"""UnHype training: a CLIP-guided hypernetwork that emits LoRA weights for a
frozen CogVideoX transformer (arXiv 2602.03410). At each step we sample a
forget/mapping concept pair, predict LoRA weights at two consecutive trajectory
steps s and s+1, and match the hypernet's own step (θ_{s+1} − θ_s) to a single
SGD update of the steered task loss. A retention loss keeps the hypernet output
near-zero for unrelated concepts."""

import os
import random
import statistics
from dataclasses import dataclass, field

import mlflow
import wandb
import torch
import torch.nn.functional as F
import pandas as pd
from diffusers import CogVideoXPipeline
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from tqdm.auto import tqdm

from zml.unlearn.eval import EvalPrompt, evaluate
from zml.unlearn.metrics_log import MetricsRecorder
from zml.unlearn.unhype_modules import (
    Hypernetwork,
    apply_hypernet_output,
    clear_hypernet_output,
    disable_hyper_adapters,
    replace_with_hyper_lora,
)
from zml.utils import set_seed


DEFAULT_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]

# Latent shape constants — match the existing CogVideoX unlearning scripts.
BATCH_SIZE = 1
NUM_CHANNELS = 16
NUM_FRAMES = 13
LATENT_HEIGHT = 60
LATENT_WIDTH = 90
NUM_INFERENCE_STEPS = 50


@dataclass
class Config:
    model_id: str
    clip_model_id: str
    target_mapping_path: str
    retain_prompts_path: str
    control_concept_prompts: str
    control_related_prompts: str
    control_unrelated_prompts: str
    lora_rank: int
    lora_alpha: float
    num_unlearning_steps: int
    simulated_lr: float
    negative_guidance_scale: float
    removal_weight: float
    retain_weight: float
    hypernet_hidden_dim: int
    hypernet_num_layers: int
    hypernet_step_embedding_dim: int
    learning_rate: float
    steps: int
    save_interval: int
    eval_num_prompts: int
    eval_inference_steps: int
    output_dir: str
    lora_target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    global_seed: int | None = None
    metrics_log_interval: int = 50  # steps per flushed train-window row in summary.json


def _load_target_mapping(path: str) -> list[tuple[str, str]]:
    df = pd.read_csv(path)
    return list(zip(df["target"].tolist(), df["mapping"].tolist()))


def _load_prompts_csv(path: str, column: str = "prompt") -> list[str]:
    return pd.read_csv(path)[column].tolist()


def _load_eval_prompts(path: str) -> list[EvalPrompt]:
    df = pd.read_csv(path)
    return [EvalPrompt(prompt=row["prompt"], seed=int(row["seed"])) for _, row in df.iterrows()]


def _tensor_norm(t: torch.Tensor) -> float:
    return float(t.detach().float().norm())


def main(config: Config) -> None:
    if config.global_seed is not None:
        set_seed(config.global_seed)

    target_mapping = _load_target_mapping(config.target_mapping_path)
    retain_prompts = _load_prompts_csv(config.retain_prompts_path)
    control_concept = _load_eval_prompts(config.control_concept_prompts)
    control_related = _load_eval_prompts(config.control_related_prompts)
    control_unrelated = _load_eval_prompts(config.control_unrelated_prompts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    pipe = CogVideoXPipeline.from_pretrained(config.model_id, torch_dtype=dtype).to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    transformer = pipe.transformer
    transformer.eval()
    transformer.requires_grad_(False)
    # The removal-loss target -η∇_{θ_s}ℒ_task is detached (see below), so the
    # task-loss forward graph is freed before backward and we only ever do a
    # first-order backward. That makes gradient checkpointing safe here — and
    # it is critical to fit the 5B transformer's activations in VRAM.
    # diffusers' default checkpointing is non-reentrant, which correctly routes
    # autograd.grad back to the closure-provided LoRA tensors.
    transformer.enable_gradient_checkpointing()

    hyper_modules, lora_shapes = replace_with_hyper_lora(
        transformer,
        target_module_names=config.lora_target_modules,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
    )
    transformer = transformer.to(device, dtype=dtype)
    pipe.transformer = transformer
    print(f"Replaced {len(hyper_modules)} Linear layers with HyperLoRALinear.")

    clip_tokenizer = CLIPTokenizer.from_pretrained(config.clip_model_id)
    clip_text_model = CLIPTextModelWithProjection.from_pretrained(config.clip_model_id).to(device)
    clip_text_model.eval()
    clip_text_model.requires_grad_(False)
    clip_dim = clip_text_model.config.projection_dim

    hypernet = Hypernetwork(
        clip_dim=clip_dim,
        lora_shapes=lora_shapes,
        rank=config.lora_rank,
        hidden_dim=config.hypernet_hidden_dim,
        num_layers=config.hypernet_num_layers,
        step_embedding_dim=config.hypernet_step_embedding_dim,
        max_step=config.num_unlearning_steps,
    ).to(device)
    n_hypernet_params = sum(p.numel() for p in hypernet.parameters())
    print(f"Hypernetwork: {n_hypernet_params:,} params, flat output dim {hypernet.total_output:,}")

    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=config.learning_rate)

    def encode_t5(prompt: str) -> torch.Tensor:
        embeds, _ = pipe.encode_prompt(prompt=prompt, do_classifier_free_guidance=False)
        return embeds.to(device, dtype=dtype)

    def encode_clip(prompt: str) -> torch.Tensor:
        tokens = clip_tokenizer(
            prompt, padding="max_length", truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = clip_text_model(**tokens)
        return out.text_embeds.to(torch.float32)

    def hypernet_predict(c_clip: torch.Tensor, s: int) -> torch.Tensor:
        s_tensor = torch.tensor([s], device=device, dtype=torch.float32)
        return hypernet(c_clip, s_tensor).squeeze(0)

    def apply_flat(flat: torch.Tensor) -> None:
        apply_hypernet_output(hyper_modules, hypernet.decode(flat))

    scheduler = pipe.scheduler
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    latent_shape = (BATCH_SIZE, NUM_CHANNELS, NUM_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    S = config.num_unlearning_steps

    recorder = MetricsRecorder(
        output_dir=config.output_dir,
        run_name=os.path.basename(config.output_dir.rstrip("/")) or "unhype",
        config={
            "method": "unhype",
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "num_unlearning_steps": S,
            "simulated_lr": config.simulated_lr,
            "negative_guidance_scale": config.negative_guidance_scale,
            "removal_weight": config.removal_weight,
            "retain_weight": config.retain_weight,
            "learning_rate": config.learning_rate,
            "steps": config.steps,
        },
        flush_interval=config.metrics_log_interval,
    )

    print("Starting UnHype training...")
    pbar = tqdm(range(config.steps))
    for step in pbar:
        target_prompt, mapping_prompt = random.choice(target_mapping)
        retain_prompt = random.choice(retain_prompts)
        s = random.randint(0, S - 1)

        c_target_clip = encode_clip(target_prompt)
        c_retain_clip = encode_clip(retain_prompt)

        theta_s = hypernet_predict(c_target_clip, s)
        theta_s_plus_1 = hypernet_predict(c_target_clip, s + 1)

        apply_flat(theta_s)

        with torch.no_grad():
            c_target_t5 = encode_t5(target_prompt)
            c_mapping_t5 = encode_t5(mapping_prompt)

        t_idx = random.randint(1, NUM_INFERENCE_STEPS - 1)
        t = scheduler.timesteps[t_idx]
        timesteps = t.unsqueeze(0).expand(BATCH_SIZE).to(device)
        latents = torch.randn(latent_shape, device=device, dtype=dtype)

        with torch.no_grad():
            for ts in scheduler.timesteps:
                if ts <= t:
                    break
                ts_batch = ts.unsqueeze(0).expand(BATCH_SIZE).to(device)
                model_in = latents.permute(0, 2, 1, 3, 4)
                noise_pred = transformer(
                    hidden_states=model_in,
                    encoder_hidden_states=c_target_t5,
                    timestep=ts_batch,
                ).sample.permute(0, 2, 1, 3, 4)
                latents = scheduler.step(noise_pred, ts, latents).prev_sample

        model_input = latents.permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            with disable_hyper_adapters(transformer):
                eps_target_concept = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=c_target_t5,
                    timestep=timesteps,
                ).sample
                eps_mapping_concept = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=c_mapping_t5,
                    timestep=timesteps,
                ).sample
        eps_steered_target = eps_mapping_concept - config.negative_guidance_scale * (
            eps_target_concept - eps_mapping_concept
        )

        eps_pred = transformer(
            hidden_states=model_input,
            encoder_hidden_states=c_target_t5,
            timestep=timesteps,
        ).sample
        loss_task = F.mse_loss(eps_pred.float(), eps_steered_target.float())

        # Gradient matching (Hypernet Fields): -η∇_{θ_s}ℒ_task is a fixed target,
        # so create_graph=False (no second-order graph) and we detach it. This
        # also frees the expensive task-loss forward graph before backward.
        # φ is still optimized via predicted_step = θ_{s+1} − θ_s below.
        grad_theta = torch.autograd.grad(loss_task, theta_s, create_graph=False)[0]
        target_step = (-config.simulated_lr * grad_theta).detach()
        predicted_step = theta_s_plus_1 - theta_s
        loss_remove = F.mse_loss(predicted_step.float(), target_step.float())

        theta_retain_s = hypernet_predict(c_retain_clip, s)
        theta_retain_0 = hypernet_predict(c_retain_clip, 0)
        loss_retain = F.mse_loss(theta_retain_s.float(), theta_retain_0.float())

        loss_total = config.removal_weight * loss_remove + config.retain_weight * loss_retain

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        clear_hypernet_output(hyper_modules)

        metrics = {
            "train/loss_task": float(loss_task.detach()),
            "train/loss_remove": float(loss_remove.detach()),
            "train/loss_retain": float(loss_retain.detach()),
            "train/loss_total": float(loss_total.detach()),
            # Diagnostics: is the trajectory leaving the origin, and how strong
            # is the steering signal that drives the whole task gradient?
            "train/theta_s_norm": _tensor_norm(theta_s),
            "train/predicted_step_norm": _tensor_norm(predicted_step),
            "train/target_step_norm": _tensor_norm(target_step),
            "train/grad_theta_norm": _tensor_norm(grad_theta),
            "train/steering_norm": _tensor_norm(eps_target_concept - eps_mapping_concept),
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)
        wandb.log(metrics, step=step)
        recorder.log_train(step, metrics)
        pbar.set_description(
            f"task={metrics['train/loss_task']:.4f} rem={metrics['train/loss_remove']:.4e} ret={metrics['train/loss_retain']:.4e}"
        )

        if (step + 1) % config.save_interval == 0:
            ckpt_dir = os.path.join(config.output_dir, f"unhype_step{step + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(hypernet.state_dict(), os.path.join(ckpt_dir, "hypernet.pt"))
            print(f"Checkpoint saved to: {ckpt_dir}")

            # Endpoint adapter magnitude on the *actual* eval prompts. If this is
            # ~0, the hypernet emits a near-empty adapter on the (long) control
            # prompts it never saw in training -> generations match the base model.
            s_endpoint = torch.tensor([S], device=device, dtype=torch.float32)
            with torch.no_grad():
                theta_S_norms = [
                    _tensor_norm(hypernet(encode_clip(ep.prompt), s_endpoint).squeeze(0))
                    for ep in control_concept[: config.eval_num_prompts]
                ]
            mean_theta_S_norm = sum(theta_S_norms) / len(theta_S_norms)
            # Spread across prompts: ~0 ⇒ the hypernet ignores its conditioning.
            std_theta_S_norm = statistics.pstdev(theta_S_norms) if len(theta_S_norms) > 1 else 0.0
            mlflow.log_metric("eval/theta_S_norm_concept", mean_theta_S_norm, step=step + 1)
            wandb.log({"eval/theta_S_norm_concept": mean_theta_S_norm}, step=step + 1)

            def prepare_for_prompt(prompt: str) -> None:
                with torch.no_grad():
                    c_clip = encode_clip(prompt)
                    flat = hypernet(
                        c_clip,
                        torch.tensor([S], device=device, dtype=torch.float32),
                    ).squeeze(0)
                    apply_flat(flat)

            eval_metrics = evaluate(
                pipe, transformer, config, step + 1,
                control_concept, control_related, control_unrelated,
                prepare_for_prompt=prepare_for_prompt,
            )
            clear_hypernet_output(hyper_modules)

            recorder.log_eval(step + 1, {
                "theta_S_norm_concept_mean": mean_theta_S_norm,
                "theta_S_norm_concept_std": std_theta_S_norm,
                "scores": {
                    set_name: {
                        "fire_detection_rate": s["fire_detection_rate"],
                        "clip_score_mean": s["clip_score_mean"],
                        "dover_technical_mean": s["dover_technical_mean"],
                        "dover_aesthetic_mean": s["dover_aesthetic_mean"],
                    }
                    for set_name, s in eval_metrics.items()
                },
            })

    recorder.close()
    print("UnHype training complete.")
