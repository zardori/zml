"""``frame_replace`` unlearning: supervised v-prediction fine-tuning toward edited latents.

Idea: for a fire prompt the model often produces a clip where fire appears only in some frames.
A precompute step (``zml/precompute/frame_replace_precompute.py``) replaces the fire-containing
latent frames with the nearest fire-free ("donor") frame from the same clip, yielding an edited
clean latent ``x0_edited``. Here we fine-tune a PEFT LoRA so the fire prompt maps toward that
fireless version of the model's own output.

Unlike the ESD-family methods this has no teacher / CFG / negative guidance — it is plain
supervised diffusion training: noise the target, predict velocity, MSE against the true velocity.
"""

import json
import os
import random
from dataclasses import dataclass

import mlflow
import pandas as pd
import wandb
import torch
import torch.nn.functional as F
from diffusers import CogVideoXPipeline
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from zml.unlearn.eval import EvalPrompt, evaluate
from zml.unlearn.metrics_log import MetricsRecorder
from zml.utils import set_seed

# Latent geometry for CogVideoX-5b (see unhype.py). RoPE temporal size is the latent frame count.
NUM_CHANNELS = 16
NUM_LATENT_FRAMES = 13
LATENT_HEIGHT = 60
LATENT_WIDTH = 90
EXPECTED_LATENT_SHAPE = (1, NUM_CHANNELS, NUM_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)

LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
DTYPE = torch.bfloat16


@dataclass
class Config:
    model_id: str
    metadata_file: str  # frame_replace metadata.json (prompt, seed, latent_path, ...)
    latents_dir: str  # directory of x0_edited .pt files
    control_concept_prompts: str
    control_related_prompts: str
    control_unrelated_prompts: str
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    steps: int
    save_interval: int
    learning_rate: float
    output_dir: str
    eval_num_prompts: int
    eval_inference_steps: int
    timestep_min: int = 0  # SFT samples raw train timesteps uniformly in [min, max)
    timestep_max: int = 1000
    num_frames: int = 49  # generation geometry (pixel frames)
    height: int = 480  # pixel height — used for rotary embeddings
    width: int = 720  # pixel width
    global_seed: int | None = None
    disable_mlflow: bool = False
    metrics_log_interval: int = 50


def _load_eval_prompts(path: str) -> list[EvalPrompt]:
    df = pd.read_csv(path)
    return [EvalPrompt(prompt=row["prompt"], seed=int(row["seed"])) for _, row in df.iterrows()]


def main(config: Config) -> None:
    if config.global_seed is not None:
        set_seed(config.global_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    control_concept = _load_eval_prompts(config.control_concept_prompts)
    control_related = _load_eval_prompts(config.control_related_prompts)
    control_unrelated = _load_eval_prompts(config.control_unrelated_prompts)

    with open(config.metadata_file) as f:
        metadata: list[dict] = json.load(f)
    if not metadata:
        raise ValueError(f"No entries in {config.metadata_file}; precompute produced no targets.")

    pipe = CogVideoXPipeline.from_pretrained(config.model_id, torch_dtype=DTYPE).to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    scheduler = pipe.scheduler
    assert scheduler.config.prediction_type == "v_prediction", (
        f"Expected v_prediction scheduler, got {scheduler.config.prediction_type!r}"
    )
    expected_scaling = float(pipe.vae.config.scaling_factor)
    metadata_scaling = float(metadata[0].get("scaling_factor", expected_scaling))
    assert abs(metadata_scaling - expected_scaling) < 1e-6, (
        f"Latents were built with scaling_factor {metadata_scaling}, model uses {expected_scaling}."
    )

    transformer = pipe.transformer
    transformer.train()
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate)

    # Cache one T5 embedding per unique prompt (CFG-free; we handle no guidance here).
    prompt_emb_cache: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for prompt in {entry["prompt"] for entry in metadata}:
            embeds, _ = pipe.encode_prompt(prompt=prompt, do_classifier_free_guidance=False)
            prompt_emb_cache[prompt] = embeds.to(device, dtype=DTYPE)

    pipe.transformer = transformer

    # Rotary embeddings depend only on the fixed latent geometry, so build them once.
    # The transformer does NOT compute these internally — eval generates with RoPE, so training
    # must use it too or the LoRA learns to correct a mismatched positional regime.
    image_rotary_emb = (
        pipe._prepare_rotary_positional_embeddings(config.height, config.width, NUM_LATENT_FRAMES, device)
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )

    recorder = MetricsRecorder(
        output_dir=config.output_dir,
        run_name=os.path.basename(config.output_dir.rstrip("/")) or "frame_replace",
        config={
            "method": "frame_replace",
            "model_id": config.model_id,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "learning_rate": config.learning_rate,
            "steps": config.steps,
            "save_interval": config.save_interval,
            "timestep_min": config.timestep_min,
            "timestep_max": config.timestep_max,
            "num_targets": len(metadata),
            "eval_num_prompts": config.eval_num_prompts,
            "eval_inference_steps": config.eval_inference_steps,
            "global_seed": config.global_seed,
        },
        flush_interval=config.metrics_log_interval,
    )

    print(f"Starting frame_replace SFT over {len(metadata)} edited targets...")
    pbar = tqdm(range(config.steps))
    for step in pbar:
        entry = random.choice(metadata)
        x0 = torch.load(
            os.path.join(config.latents_dir, entry["latent_path"]), map_location=device
        ).to(dtype=DTYPE)  # (B, C, F, H, W), scaled latent space
        assert x0.shape == EXPECTED_LATENT_SHAPE, f"unexpected target shape {x0.shape}"
        concept_emb = prompt_emb_cache[entry["prompt"]]

        # Uniform integer timesteps are correct here: CogVideoX's SNR shift lives in the
        # scheduler's alphas_cumprod (not the timestep grid), so add_noise/get_velocity map each
        # sampled t through the shifted noise levels automatically. No need to match the inference
        # timestep grid — that grid is ~evenly spaced in index, so sampling from it would give the
        # same noise-level distribution. To upweight high-noise steps (where the concept is
        # decided), raise timestep_min rather than reshaping the sampler.
        t = torch.randint(config.timestep_min, config.timestep_max, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        x_t = scheduler.add_noise(x0, noise, t)
        v_target = scheduler.get_velocity(x0, noise, t)  # (B, C, F, H, W)

        v_pred = transformer(
            hidden_states=x_t.permute(0, 2, 1, 3, 4),  # -> (B, F, C, H, W)
            encoder_hidden_states=concept_emb,
            timestep=t,
            image_rotary_emb=image_rotary_emb,
        ).sample  # (B, F, C, H, W)

        loss = F.mse_loss(v_pred.float(), v_target.permute(0, 2, 1, 3, 4).float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recorder.log_train(step, {"train/loss": loss.item(), "train/timestep": float(t[0].item())})
        if not config.disable_mlflow:
            mlflow.log_metric("train/loss", loss.item(), step=step)
        wandb.log({"train/loss": loss.item()}, step=step)
        pbar.set_description(f"loss={loss.item():.4f}")

        if (step + 1) % config.save_interval == 0:
            ckpt_dir = os.path.join(config.output_dir, f"frame_replace_lora_step{step + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            transformer.save_pretrained(ckpt_dir)
            print(f"Checkpoint saved to: {ckpt_dir}")
            eval_metrics = evaluate(
                pipe, transformer, config, step + 1,
                control_concept, control_related, control_unrelated,
                log_mlflow=not config.disable_mlflow,
            )
            recorder.log_eval(step + 1, {
                "scores": {
                    set_name: {
                        "fire_detection_rate": s["fire_detection_rate"],
                        "clip_score_mean": s["clip_score_mean"],
                        "colorfulness_mean": s["colorfulness_mean"],
                        "dover_technical_mean": s["dover_technical_mean"],
                        "dover_aesthetic_mean": s["dover_aesthetic_mean"],
                    }
                    for set_name, s in eval_metrics.items()
                },
            })

    recorder.close()
    print("frame_replace training complete.")
