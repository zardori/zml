import os
import random
from dataclasses import dataclass

import mlflow
import wandb
import torch
import torch.nn.functional as F
from diffusers import CogVideoXPipeline
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
import pandas as pd

from zml.unlearn.eval import EvalPrompt, evaluate
from zml.unlearn.metrics_log import MetricsRecorder
from zml.utils import set_seed

NORM_EPS = 1e-8


def _per_sample_norm(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """L2 norm over every non-batch dim, returning one scalar per sample (shape [B])."""
    return torch.linalg.vector_norm(x, dim=dims)


def _grad_norm(parameters) -> float:
    """Total L2 norm of all parameter gradients (blow-up / dead-gradient diagnostic)."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in grads])))


@dataclass
class Config:
    model_id: str
    prompts_path: str
    control_concept_prompts: str
    control_related_prompts: str
    control_unrelated_prompts: str
    lora_rank: int
    lora_alpha: float
    negative_guidance_scale: float
    steps: int
    save_interval: int
    learning_rate: float
    lora_dropout: float
    output_dir: str
    eval_num_prompts: int
    eval_inference_steps: int
    global_seed: int | None = None
    disable_mlflow: bool = False
    metrics_log_interval: int = 50  # steps per flushed train-window row in summary.json


def main(config: Config):
    if config.global_seed is not None:
        set_seed(config.global_seed)

    data = pd.read_csv(config.prompts_path)
    CONCEPT_PROMPTS = data["prompt"].tolist()

    with open(config.control_concept_prompts) as f:
        CONTROL_CONCEPT_PROMPTS = [EvalPrompt(p, 42 + i) for i, p in enumerate(l.strip() for l in f if l.strip())]
    with open(config.control_related_prompts) as f:
        CONTROL_RELATED_PROMPTS = [EvalPrompt(p, 42 + i) for i, p in enumerate(l.strip() for l in f if l.strip())]
    with open(config.control_unrelated_prompts) as f:
        CONTROL_UNRELATED_PROMPTS = [EvalPrompt(p, 42 + i) for i, p in enumerate(l.strip() for l in f if l.strip())]

    # @title 2. Setup Configuration
    # Using CogVideoX-2b (Fits on approx 12-16GB VRAM with optimizations)
    MODEL_ID = config.model_id
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 # CogVideoX works best with bf16

    # Load Pipeline
    pipe = CogVideoXPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE
    ).to(DEVICE)

    # Enable memory savings
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 1. Extract the Transformer (The noise predictor)
    transformer = pipe.transformer
    transformer.train()
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing() # Critical for VRAM

    # 2. Define LoRA Config for Transformer
    # Targeting the attention projections in the DiT
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=config.lora_dropout,
        bias="none",
    )

    # 3. Inject LoRA
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Wire the PEFT-wrapped transformer back into the pipeline for eval generation
    pipe.transformer = transformer

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate)

    # Helper to encode text specific to CogVideoX
    def encode_prompt(pipe, prompt):
        # CogVideoX uses T5 and requires specific embedding handling
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False # We handle CFG manually in loop
        )
        return prompt_embeds.to(DEVICE, dtype=DTYPE)

    print("Starting ESD Training for Video...")

    scheduler = pipe.scheduler

    # Latent setup for CogVideoX
    # Initial Generation Shape: [Batch, Channels, Frames, Height, Width]
    batch_size = 1
    num_channels = 16
    num_frames = 13
    height = 60
    width = 90

    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps)

    recorder = MetricsRecorder(
        output_dir=config.output_dir,
        run_name=os.path.basename(config.output_dir.rstrip("/")) or "esd_normalized",
        config={
            "method": "esd_normalized",
            "model_id": config.model_id,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "negative_guidance_scale": config.negative_guidance_scale,
            "learning_rate": config.learning_rate,
            "steps": config.steps,
            "save_interval": config.save_interval,
            "eval_num_prompts": config.eval_num_prompts,
            "eval_inference_steps": config.eval_inference_steps,
            "global_seed": config.global_seed,
        },
        flush_interval=config.metrics_log_interval,
    )

    pbar = tqdm(range(config.steps))

    for step in pbar:
        CONCEPT_PROMPT = random.choice(CONCEPT_PROMPTS)
        with torch.no_grad():
            concept_emb = encode_prompt(pipe, CONCEPT_PROMPT)
            null_emb = encode_prompt(pipe, "")

        optimizer.zero_grad()

        # 1. Sample a random denoising step t (not the very first step)
        t_idx = random.randint(1, num_inference_steps - 1)
        t = scheduler.timesteps[t_idx]
        timesteps = t.unsqueeze(0).expand(batch_size).to(DEVICE)

        # 2. Start from pure noise x_T [B, C, F, H, W]
        latents = torch.randn(
            (batch_size, num_channels, num_frames, height, width),
            device=DEVICE,
            dtype=DTYPE
        )

        # 3. Partially denoise with the student model from T down to t
        with torch.no_grad():
            for ts in scheduler.timesteps:
                if ts <= t:
                    break
                model_input = latents.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W] -> [B, F, C, H, W]
                noise_pred = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=concept_emb,
                    timestep=ts.unsqueeze(0).expand(batch_size).to(DEVICE),
                ).sample.permute(0, 2, 1, 3, 4)  # back to [B, C, F, H, W]
                latents = scheduler.step(noise_pred, ts, latents).prev_sample

        # Convert to transformer format [B, F, C, H, W] for ESD loss computation
        model_input = latents.permute(0, 2, 1, 3, 4)

        # -----------------------------------------------------------
        # TEACHER STEP (Frozen DiT)
        # -----------------------------------------------------------
        with torch.no_grad():
            with transformer.disable_adapter():
                # Unconditional
                model_pred_uncond = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=null_emb,
                    timestep=timesteps,
                ).sample

                # Conditional (Concept)
                model_pred_text = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=concept_emb,
                    timestep=timesteps,
                ).sample

                # ESD Target Calculation
                direction = model_pred_text - model_pred_uncond
                norm_dims = tuple(range(1, direction.ndim))
                direction_norm = torch.linalg.vector_norm(direction, dim=norm_dims, keepdim=True)
                uncond_norm = torch.linalg.vector_norm(model_pred_uncond, dim=norm_dims, keepdim=True)
                direction = direction * (uncond_norm / (direction_norm + NORM_EPS))
                target = model_pred_uncond - config.negative_guidance_scale * direction

        # -----------------------------------------------------------
        # STUDENT STEP (Trainable LoRA)
        # -----------------------------------------------------------
        model_pred = transformer(
            hidden_states=model_input,
            encoder_hidden_states=concept_emb,
            timestep=timesteps,
        ).sample

        # Loss
        loss = F.mse_loss(model_pred.float(), target.float())

        loss.backward()
        # grad_norm read after backward (optimizer.step does not clear grads).
        grad_norm = _grad_norm(transformer.parameters())
        optimizer.step()

        with torch.no_grad():
            # The ESD target sits `target_shift` away from the frozen concept prediction
            # (the distance the student must travel); `student_drift` is how far the student
            # has actually moved off it. Their ratio is a scale-free erasure-progress signal:
            # ~0 at init (LoRA ≈ identity) and ->1 once the student matches the target.
            target_shift = _per_sample_norm(target - model_pred_text, norm_dims)
            student_drift = _per_sample_norm(model_pred.detach() - model_pred_text, norm_dims)
            erase_progress = (student_drift / (target_shift + NORM_EPS)).mean().item()
            # Frozen-teacher sanity check: relative size of the concept's perturbation to the
            # unconditional prediction. Near 0 means the prompt carries no separable signal.
            concept_strength = (direction_norm / (uncond_norm + NORM_EPS)).mean().item()

        recorder.log_train(step, {
            "train/loss": loss.item(),
            "train/erase_progress": erase_progress,
            "train/student_drift": student_drift.mean().item(),
            "train/target_shift": target_shift.mean().item(),
            "train/concept_strength": concept_strength,
            "train/grad_norm": grad_norm,
            "train/timestep": float(t.item()),
        })

        if not config.disable_mlflow:
            mlflow.log_metric("train/loss", loss.item(), step=step)
        wandb.log({"train/loss": loss.item()}, step=step)
        pbar.set_description(f"loss={loss.item():.4f} progress={erase_progress:.3f}")

        if (step + 1) % config.save_interval == 0:
            lora_output_dir = os.path.join(config.output_dir, f"cogvideox_erasure_lora_nudity_step{step + 1}")
            os.makedirs(lora_output_dir, exist_ok=True)
            transformer.save_pretrained(lora_output_dir)
            print(f"Checkpoint saved to: {lora_output_dir}")
            eval_metrics = evaluate(pipe, transformer, config, step + 1,
                     CONTROL_CONCEPT_PROMPTS, CONTROL_RELATED_PROMPTS, CONTROL_UNRELATED_PROMPTS,
                     log_mlflow=not config.disable_mlflow)
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
    print("Training Complete.")

