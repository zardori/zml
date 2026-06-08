"""UnHype training: a CLIP-guided hypernetwork that emits LoRA weights for a
frozen CogVideoX transformer (arXiv 2602.03410). At each step we sample a
forget/mapping concept pair, predict LoRA weights at two consecutive trajectory
steps s and s+1, and match the hypernet's own step (θ_{s+1} − θ_s) to a single
SGD update of the steered task loss. A retention loss keeps the hypernet output
near-zero for unrelated concepts.

This module owns the shared setup (``UnhypeContext`` / ``build_context``) and the
real "online" method. The debug-only control modes (``distill``, ``static_apply``)
live in ``unhype_debug.py`` and are dispatched to from ``main`` by ``target_mode``."""

import os
import random
import statistics
from collections.abc import Callable
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
    HyperLoRALinear,
    Hypernetwork,
    LoRAShape,
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
LATENT_SHAPE = (BATCH_SIZE, NUM_CHANNELS, NUM_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)


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
    remove_loss_type: str = "mse"  # "mse" | "cosine"
    remove_magnitude_weight: float = 1.0  # weight of the ‖·‖-matching term in the cosine variant
    target_grad_batch_size: int = 1  # # of (timestep, latent) samples averaged into the removal target
    target_mode: str = "online"  # "online" | "distill" | "static_apply"
    distill_adapter_dir: str = ""  # PEFT adapter dir for distill mode (the endpoint θ* to reproduce)
    target_prompt_batch_size: int = 1  # # of prompt pairs averaged into the online removal target
    optimizer: str = "adamw"  # "adamw" | "adafactor" — adafactor's factored state fits rank-8 hypernet


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


@dataclass
class UnhypeContext:
    """Everything built once at startup and shared by all target modes. The online
    loop and the debug controls (``unhype_debug.py``) consume this instead of the
    sprawling closure soup that used to live inside ``main``."""

    pipe: CogVideoXPipeline
    transformer: torch.nn.Module
    scheduler: object
    hypernet: Hypernetwork
    hyper_modules: list[HyperLoRALinear]
    lora_shapes: list[LoRAShape]
    optimizer: torch.optim.Optimizer
    clip_tokenizer: CLIPTokenizer
    clip_text_model: CLIPTextModelWithProjection
    recorder: MetricsRecorder
    device: str
    dtype: torch.dtype
    S: int
    target_mapping: list[tuple[str, str]]
    retain_prompts: list[str]
    control_concept: list[EvalPrompt]
    control_related: list[EvalPrompt]
    control_unrelated: list[EvalPrompt]

    def encode_t5(self, prompt: str) -> torch.Tensor:
        embeds, _ = self.pipe.encode_prompt(prompt=prompt, do_classifier_free_guidance=False)
        return embeds.to(self.device, dtype=self.dtype)

    def encode_clip(self, prompt: str) -> torch.Tensor:
        tokens = self.clip_tokenizer(
            prompt, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.clip_text_model(**tokens)
        return out.text_embeds.to(torch.float32)

    def hypernet_predict(self, c_clip: torch.Tensor, s: int) -> torch.Tensor:
        s_tensor = torch.tensor([s], device=self.device, dtype=torch.float32)
        return self.hypernet(c_clip, s_tensor).squeeze(0)

    def apply_flat(self, flat: torch.Tensor) -> None:
        apply_hypernet_output(self.hyper_modules, self.hypernet.decode(flat))

    def move_diffusion_stack(self, target_device: str) -> None:
        """Move transformer/T5/VAE between CPU and GPU. Distill training needs only the
        hypernet + CLIP, so the ~21 GB diffusion stack is offloaded to CPU during the loop
        and brought back only for the periodic eval — the rank-8 hypernet's ~34 GB
        weight+grad otherwise overflows the GPU at backward(). clip_text_model and hypernet
        are separate from pipe, so they stay on GPU."""
        self.pipe.to(target_device)
        torch.cuda.empty_cache()


def build_context(config: Config) -> UnhypeContext:
    """Load the model, wrap target Linears with HyperLoRA, build the hypernet,
    optimizer, prompts and metrics recorder. Shared by online + debug modes."""
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

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(hypernet.parameters(), lr=config.learning_rate)
    elif config.optimizer == "adafactor":
        # Factored second moment + no first-moment buffer ⇒ ~0 optimizer-state memory, vs AdamW's
        # 2x params (~34 GB at rank 8). Required to fit the rank-8 hypernet's ~4.2B-param output
        # layer alongside the 5B transformer on a 95 GB GPU (the distill control).
        optimizer = torch.optim.Adafactor(hypernet.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer!r} (expected 'adamw' | 'adafactor')")

    scheduler = pipe.scheduler
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    recorder = MetricsRecorder(
        output_dir=config.output_dir,
        run_name=os.path.basename(config.output_dir.rstrip("/")) or "unhype",
        config={
            "method": "unhype",
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "num_unlearning_steps": config.num_unlearning_steps,
            "simulated_lr": config.simulated_lr,
            "negative_guidance_scale": config.negative_guidance_scale,
            "removal_weight": config.removal_weight,
            "retain_weight": config.retain_weight,
            "learning_rate": config.learning_rate,
            "steps": config.steps,
            "target_grad_batch_size": config.target_grad_batch_size,
            "target_mode": config.target_mode,
            "target_prompt_batch_size": config.target_prompt_batch_size,
            "optimizer": config.optimizer,
        },
        flush_interval=config.metrics_log_interval,
    )

    return UnhypeContext(
        pipe=pipe,
        transformer=transformer,
        scheduler=scheduler,
        hypernet=hypernet,
        hyper_modules=hyper_modules,
        lora_shapes=lora_shapes,
        optimizer=optimizer,
        clip_tokenizer=clip_tokenizer,
        clip_text_model=clip_text_model,
        recorder=recorder,
        device=device,
        dtype=dtype,
        S=config.num_unlearning_steps,
        target_mapping=target_mapping,
        retain_prompts=retain_prompts,
        control_concept=control_concept,
        control_related=control_related,
        control_unrelated=control_unrelated,
    )


def compute_removal_loss(
    predicted_step: torch.Tensor,
    target_step: torch.Tensor,
    loss_type: str,
    magnitude_weight: float,
) -> tuple[torch.Tensor, float, float]:
    """Removal loss + (direction, magnitude) diagnostics.

    "mse"    – original ‖pred − target‖²; its gradient is dominated by ‖pred‖ when the two
               scales differ, so a large predicted step is just shrunk toward zero (the
               trajectory collapses to constant-in-s) instead of being steered.
    "cosine" – (1 − cos(pred, target)) + magnitude_weight·(‖pred‖ − ‖target‖)²; the direction
               term aligns the step with the erasure gradient without driving its magnitude to 0.
    """
    pred = predicted_step.float()
    tgt = target_step.float()
    if loss_type == "mse":
        loss = F.mse_loss(pred, tgt)
        return loss, float(loss.detach()), 0.0
    if loss_type == "cosine":
        direction = 1.0 - F.cosine_similarity(pred, tgt, dim=0)
        magnitude = (pred.norm() - tgt.norm()).pow(2)
        loss = direction + magnitude_weight * magnitude
        return loss, float(direction.detach()), float(magnitude.detach())
    raise ValueError(f"Unknown remove_loss_type: {loss_type!r}")


def _rollout_to_timesteps(
    transformer,
    scheduler,
    encoder_hidden_states: torch.Tensor,
    init_latents: torch.Tensor,
    target_timesteps: list[torch.Tensor],
    batch_size: int,
) -> list[torch.Tensor]:
    """Partial-denoise ``init_latents`` once, snapshotting the latent state at each target timestep.

    ``target_timesteps`` must be sorted descending (high noise → low). Returns one latent snapshot per
    target, in the same order. The snapshots share the rollout prefix, so K targets cost a single
    rollout instead of K — this is what makes averaging the removal target over many timesteps cheap
    (the rollout, not the loss forwards, dominates step cost). Matches the original break-before-step
    convention: the snapshot for ``t`` is the latent just before the first scheduler step at ``ts ≤ t``.
    """
    snapshots: list[torch.Tensor] = []
    latents = init_latents
    idx = 0
    with torch.no_grad():
        for ts in scheduler.timesteps:
            while idx < len(target_timesteps) and bool(ts <= target_timesteps[idx]):
                snapshots.append(latents)
                idx += 1
            if idx >= len(target_timesteps):
                break
            ts_batch = ts.unsqueeze(0).expand(batch_size).to(latents.device)
            model_in = latents.permute(0, 2, 1, 3, 4)
            noise_pred = transformer(
                hidden_states=model_in,
                encoder_hidden_states=encoder_hidden_states,
                timestep=ts_batch,
            ).sample.permute(0, 2, 1, 3, 4)
            latents = scheduler.step(noise_pred, ts, latents).prev_sample
    while idx < len(target_timesteps):  # targets below the lowest scheduler timestep
        snapshots.append(latents)
        idx += 1
    return snapshots


def _esd_target_grad(
    transformer,
    apply_theta: Callable[[torch.Tensor], None],
    theta_s: torch.Tensor,
    snapshot_latents: torch.Tensor,
    timestep_value: torch.Tensor,
    c_target_t5: torch.Tensor,
    c_mapping_t5: torch.Tensor,
    negative_guidance_scale: float,
    batch_size: int,
) -> tuple[torch.Tensor, float, float]:
    """One ESD-steered task-loss gradient w.r.t. ``theta_s`` at a single (timestep, latent) sample.

    Re-applies ``theta_s`` so each call has a live decode graph back to it — required because we take
    several independent ``autograd.grad`` calls (one per snapshot), each of which frees its graph.
    Returns ``(grad_theta, loss_task, steering_norm)``.
    """
    apply_theta(theta_s)
    timesteps = timestep_value.unsqueeze(0).expand(batch_size).to(snapshot_latents.device)
    model_input = snapshot_latents.permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        with disable_hyper_adapters(transformer):
            eps_target_concept = transformer(
                hidden_states=model_input, encoder_hidden_states=c_target_t5, timestep=timesteps
            ).sample
            eps_mapping_concept = transformer(
                hidden_states=model_input, encoder_hidden_states=c_mapping_t5, timestep=timesteps
            ).sample
    eps_steered_target = eps_mapping_concept - negative_guidance_scale * (
        eps_target_concept - eps_mapping_concept
    )
    eps_pred = transformer(
        hidden_states=model_input, encoder_hidden_states=c_target_t5, timestep=timesteps
    ).sample
    loss_task = F.mse_loss(eps_pred.float(), eps_steered_target.float())
    grad_theta = torch.autograd.grad(loss_task, theta_s, create_graph=False)[0]
    steering_norm = _tensor_norm(eps_target_concept - eps_mapping_concept)
    return grad_theta, float(loss_task.detach()), steering_norm


def online_removal_grad(
    ctx: UnhypeContext, config: Config, theta_s: torch.Tensor, prompt_pairs: list[tuple[str, str]]
) -> tuple[torch.Tensor, float, float]:
    """Averaged ESD removal direction -∇_{θ_s}ℒ_task over a batch of prompt pairs
    and K (timestep, latent) snapshots. The diffusion timestep and the prompt are
    the two dominant variance sources; exp027 averaged only the timestep, so here
    we also average over prompts to expose the *expected* descent direction rather
    than one near-random per-step gradient. Cost is ~linear in the prompt count
    (each pair needs its own rollout), unlike timesteps which share one rollout."""
    n_timestep_samples = min(config.target_grad_batch_size, NUM_INFERENCE_STEPS - 1)
    grad_theta = torch.zeros_like(theta_s)
    loss_task_acc = 0.0
    steering_norm_acc = 0.0
    n_samples = 0
    for target_prompt, mapping_prompt in prompt_pairs:
        with torch.no_grad():
            c_target_t5 = ctx.encode_t5(target_prompt)
            c_mapping_t5 = ctx.encode_t5(mapping_prompt)
        t_indices = sorted(random.sample(range(1, NUM_INFERENCE_STEPS), n_timestep_samples))
        # scheduler.timesteps is descending, so ascending indices give descending timestep values.
        target_timesteps = [ctx.scheduler.timesteps[i] for i in t_indices]
        init_latents = torch.randn(LATENT_SHAPE, device=ctx.device, dtype=ctx.dtype)
        snapshots = _rollout_to_timesteps(
            ctx.transformer, ctx.scheduler, c_target_t5, init_latents, target_timesteps, BATCH_SIZE
        )
        for snap_latents, t_value in zip(snapshots, target_timesteps):
            grad_k, loss_k, steering_k = _esd_target_grad(
                ctx.transformer, ctx.apply_flat, theta_s, snap_latents, t_value,
                c_target_t5, c_mapping_t5, config.negative_guidance_scale, BATCH_SIZE,
            )
            grad_theta = grad_theta + grad_k
            loss_task_acc += loss_k
            steering_norm_acc += steering_k
            n_samples += 1
    grad_theta = grad_theta / n_samples
    return grad_theta, loss_task_acc / n_samples, steering_norm_acc / n_samples


def apply_optimizer_step(
    ctx: UnhypeContext,
    config: Config,
    step: int,
    s: int,
    loss_remove: torch.Tensor,
    loss_task_value: float,
    remove_metrics: dict[str, float],
    pbar: tqdm,
) -> None:
    """Retain loss + optimizer step + metrics logging — identical for online and distill,
    which differ only in how ``loss_remove``/``remove_metrics``/``s`` are produced."""
    retain_prompt = random.choice(ctx.retain_prompts)
    c_retain_clip = ctx.encode_clip(retain_prompt)
    theta_retain_s = ctx.hypernet_predict(c_retain_clip, s)
    theta_retain_0 = ctx.hypernet_predict(c_retain_clip, 0)
    loss_retain = F.mse_loss(theta_retain_s.float(), theta_retain_0.float())

    loss_total = config.removal_weight * loss_remove + config.retain_weight * loss_retain

    ctx.optimizer.zero_grad()
    loss_total.backward()
    ctx.optimizer.step()
    clear_hypernet_output(ctx.hyper_modules)

    metrics = {
        "train/loss_task": loss_task_value,
        "train/loss_remove": float(loss_remove.detach()),
        "train/loss_retain": float(loss_retain.detach()),
        "train/loss_total": float(loss_total.detach()),
        **remove_metrics,
    }
    for k, v in metrics.items():
        mlflow.log_metric(k, v, step=step)
    wandb.log(metrics, step=step)
    ctx.recorder.log_train(step, metrics)
    pbar.set_description(
        f"task={metrics['train/loss_task']:.4f} rem={metrics['train/loss_remove']:.4e} ret={metrics['train/loss_retain']:.4e}"
    )


def run_eval_checkpoint(
    ctx: UnhypeContext, config: Config, ckpt_step: int, *, offload_diffusion: bool
) -> None:
    """Save the hypernet, measure the endpoint adapter magnitude on the real eval prompts,
    and run the full video evaluation. Shared by online and distill; distill must bring the
    offloaded diffusion stack back to GPU for ``evaluate`` (``offload_diffusion=True``)."""
    ckpt_dir = os.path.join(config.output_dir, f"unhype_step{ckpt_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(ctx.hypernet.state_dict(), os.path.join(ckpt_dir, "hypernet.pt"))
    print(f"Checkpoint saved to: {ckpt_dir}")

    # Endpoint adapter magnitude on the *actual* eval prompts. If this is
    # ~0, the hypernet emits a near-empty adapter on the (long) control
    # prompts it never saw in training -> generations match the base model.
    s_endpoint = torch.tensor([ctx.S], device=ctx.device, dtype=torch.float32)
    with torch.no_grad():
        theta_S_norms = [
            _tensor_norm(ctx.hypernet(ctx.encode_clip(ep.prompt), s_endpoint).squeeze(0))
            for ep in ctx.control_concept[: config.eval_num_prompts]
        ]
    mean_theta_S_norm = sum(theta_S_norms) / len(theta_S_norms)
    # Spread across prompts: ~0 ⇒ the hypernet ignores its conditioning.
    std_theta_S_norm = statistics.pstdev(theta_S_norms) if len(theta_S_norms) > 1 else 0.0
    mlflow.log_metric("eval/theta_S_norm_concept", mean_theta_S_norm, step=ckpt_step)
    wandb.log({"eval/theta_S_norm_concept": mean_theta_S_norm}, step=ckpt_step)

    def prepare_for_prompt(prompt: str) -> None:
        with torch.no_grad():
            c_clip = ctx.encode_clip(prompt)
            flat = ctx.hypernet(c_clip, s_endpoint).squeeze(0)
            ctx.apply_flat(flat)

    if offload_diffusion:
        ctx.move_diffusion_stack(ctx.device)
    eval_metrics = evaluate(
        ctx.pipe, ctx.transformer, config, ckpt_step,
        ctx.control_concept, ctx.control_related, ctx.control_unrelated,
        prepare_for_prompt=prepare_for_prompt,
    )
    clear_hypernet_output(ctx.hyper_modules)
    if offload_diffusion:
        ctx.move_diffusion_stack("cpu")

    ctx.recorder.log_eval(ckpt_step, {
        "theta_S_norm_concept_mean": mean_theta_S_norm,
        "theta_S_norm_concept_std": std_theta_S_norm,
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


def _run_online(ctx: UnhypeContext, config: Config) -> None:
    """The real method: match the hypernet's finite-difference step θ_{s+1} − θ_s at a
    random trajectory point s to one SGD update of the ESD-steered task loss."""
    print("Starting UnHype training...")
    pbar = tqdm(range(config.steps))
    for step in pbar:
        s = random.randint(0, ctx.S - 1)
        n_pairs = min(config.target_prompt_batch_size, len(ctx.target_mapping))
        prompt_pairs = random.sample(ctx.target_mapping, n_pairs)
        c_target_clip = ctx.encode_clip(prompt_pairs[0][0])
        theta_s = ctx.hypernet_predict(c_target_clip, s)
        theta_s_plus_1 = ctx.hypernet_predict(c_target_clip, s + 1)
        # Gradient matching (Hypernet Fields): -η∇_{θ_s}ℒ_task is a fixed target
        # (detached, first-order). φ is optimized via predicted_step = θ_{s+1} − θ_s.
        grad_theta, loss_task_value, steering_norm_value = online_removal_grad(
            ctx, config, theta_s, prompt_pairs
        )
        target_step = (-config.simulated_lr * grad_theta).detach()
        predicted_step = theta_s_plus_1 - theta_s
        loss_remove, remove_direction, remove_magnitude = compute_removal_loss(
            predicted_step, target_step, config.remove_loss_type, config.remove_magnitude_weight
        )
        remove_metrics = {
            "train/loss_remove_direction": remove_direction,
            "train/loss_remove_magnitude": remove_magnitude,
            # Diagnostics: is the trajectory leaving the origin, and how strong
            # is the steering signal that drives the whole task gradient?
            "train/theta_s_norm": _tensor_norm(theta_s),
            "train/predicted_step_norm": _tensor_norm(predicted_step),
            "train/target_step_norm": _tensor_norm(target_step),
            "train/grad_theta_norm": _tensor_norm(grad_theta),
            "train/steering_norm": steering_norm_value,
        }

        apply_optimizer_step(ctx, config, step, s, loss_remove, loss_task_value, remove_metrics, pbar)

        if (step + 1) % config.save_interval == 0:
            run_eval_checkpoint(ctx, config, step + 1, offload_diffusion=False)

    ctx.recorder.close()
    print("UnHype training complete.")


def main(config: Config) -> None:
    ctx = build_context(config)
    if config.target_mode == "static_apply":
        from zml.unlearn.unhype_debug import run_static_apply
        run_static_apply(ctx, config)
    elif config.target_mode == "distill":
        from zml.unlearn.unhype_debug import run_distill
        run_distill(ctx, config)
    elif config.target_mode == "online":
        _run_online(ctx, config)
    else:
        raise ValueError(f"Unknown target_mode: {config.target_mode!r}")
