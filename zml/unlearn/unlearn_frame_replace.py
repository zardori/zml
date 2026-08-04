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

# Cosine LR decay floor, as a fraction of the initial learning_rate.
COSINE_LR_FLOOR_FRACTION = 0.05


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
    # Concept whose detector runs at eval time (see zml/benchmarks/registry.py). Selects the video
    # detector in zml/unlearn/eval.py; unrelated to the training data itself. `concept_target` names
    # the specific thing inside a concept family — the ImageNet class for concept "object".
    concept: str = "fire"
    concept_target: str | None = None
    # Retention anchor: SFT toward the base model's *unedited* preservation latents (built by
    # zml/precompute/preservation_precompute.py) to keep erasure local. Disabled when the
    # metadata file is unset, in which case the loop is plain erase-only SFT.
    retention_metadata_file: str | None = None
    retention_latents_dir: str | None = None
    retention_weight: float = 1.0
    # Drop retention anchors whose metadata `class_name` equals this. One preservation dataset can
    # then cover every class in an object-erasure protocol while each run anchors only the classes it
    # must preserve — anchoring the erased class itself would fight the erase branch directly.
    retention_exclude: str | None = None
    # Erase loss is restricted to the edited (fire) latent frames; unedited frames get this
    # weight (0.0 = hard mask). They match the base output, so weighting them in dilutes erasure.
    nonfire_frame_weight: float = 0.0
    # Space the erase MSE is computed in. "velocity" (default) reproduces exp043/044: it MSEs
    # the predicted velocity, where the fireless edit is a vanishing fraction of the target at
    # high-noise timesteps and the gradient is swamped by noise-matching. "x0" recovers the
    # predicted clean latent and MSEs that against the edited target, making the edit the full
    # supervision signal at every timestep. The retention branch always stays in velocity space.
    erase_loss_space: str = "velocity"
    # Which latent the erase branch noises to form x_t. "edited" (default) reconstructs the
    # fireless target from its own noised versions — self-consistency on the fireless manifold,
    # which exp043-045 showed never redirects the fire prompt at inference. "original" noises the
    # pre-edit fire latent (the state the model actually traverses when generating fire) and still
    # regresses toward the edited fireless target, teaching a fire->fireless denoising redirection.
    # Requires "original_latent_path" in the target metadata (precompute saves it).
    erase_input_latent: str = "edited"
    # ESD-style partial redirection for the erase branch. ``None`` (default) keeps plain SFT: the
    # student regresses *all the way* to the donor (edited) target. When set, the erase target
    # becomes ``teacher - eta*(teacher - donor) = (1-eta)*teacher + eta*donor``, where ``teacher``
    # is the frozen base model's own prediction on the same noisy fire latent+prompt (LoRA
    # disabled). The donor is not our true target — we only want to move *toward* fireless, not
    # onto the specific edited latent — so eta<1 stops the target partway and the loss can fall to
    # ~0 while the student sits at a genuine midpoint instead of overfitting the donor. eta=1
    # reproduces the plain-SFT target; eta=0 is a no-op (target == base prediction). Only the
    # erase branch uses this; the retention branch always regresses fully to its anchor latent.
    erase_esd_eta: float | None = None
    # Micro-batch size per optimizer step: each micro-step draws an independent (target,
    # timestep, noise) sample for both branches and gradients are averaged. >1 cuts the
    # batch-1 gradient noise that made exp046's eval oscillate through the erasure basin.
    gradient_accumulation_steps: int = 1
    # "constant" keeps the fixed learning_rate; "cosine" anneals it over `steps` down to
    # COSINE_LR_FLOOR_FRACTION * learning_rate so the LoRA settles instead of wandering
    # around the solution at full step size.
    lr_scheduler: str = "constant"
    # Linear ramp from ~0 to learning_rate over this many steps before lr_scheduler takes over.
    # 0 disables (LR is at full value from step 1, the original behavior). Guards against the
    # first few steps of a small-dataset LoRA fine-tune taking a full-size step before the model
    # has seen much data.
    lr_warmup_steps: int = 0
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


def _load_target_latent(latents_dir: str, latent_path: str, device: str) -> torch.Tensor:
    x0 = torch.load(os.path.join(latents_dir, latent_path), map_location=device).to(dtype=DTYPE)
    assert x0.shape == EXPECTED_LATENT_SHAPE, f"unexpected target shape {x0.shape}"
    return x0


def _fire_frame_mask(entry: dict, nonfire_weight: float, device: str) -> torch.Tensor:
    """Per-frame weights from a target's concept mask for the masked erase loss.

    Reads ``concept_latent_mask`` (the concept-agnostic key written by the split builder), falling
    back to ``fire_latent_mask`` for the original fire datasets (exp042/exp056). Edited (concept)
    frames get weight 1.0; unedited frames get ``nonfire_weight`` (0.0 hard-masks them out). Returned
    as ``(1, F, 1, 1, 1)`` to broadcast over the (B, F, C, H, W) velocity.
    """
    mask = entry.get("concept_latent_mask") or entry.get("fire_latent_mask")
    assert mask is not None, "target metadata has neither 'concept_latent_mask' nor 'fire_latent_mask'"
    assert len(mask) == NUM_LATENT_FRAMES, f"expected {NUM_LATENT_FRAMES} mask frames, got {len(mask)}"
    weights = [1.0 if is_concept else nonfire_weight for is_concept in mask]
    return torch.tensor(weights, device=device).view(1, NUM_LATENT_FRAMES, 1, 1, 1)


def _predict_x0(x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor, scheduler) -> torch.Tensor:
    """Recover the predicted clean latent from a v-prediction output.

    For v-parameterization ``v = sqrt(acp) * noise - sqrt(1 - acp) * x0`` and
    ``x_t = sqrt(acp) * x0 + sqrt(1 - acp) * noise``, which invert to
    ``x0 = sqrt(acp) * x_t - sqrt(1 - acp) * v`` with ``acp = alphas_cumprod[t]``.
    All tensors are in ``(B, F, C, H, W)`` layout; the per-sample coefficients broadcast.
    """
    acp = scheduler.alphas_cumprod.to(device=x_t.device, dtype=torch.float32)[t]
    sqrt_acp = acp.sqrt().view(-1, 1, 1, 1, 1)
    sqrt_one_minus_acp = (1.0 - acp).sqrt().view(-1, 1, 1, 1, 1)
    return sqrt_acp * x_t.float() - sqrt_one_minus_acp * v_pred.float()


def _sft_velocity_loss(
    transformer,
    scheduler,
    x0_input: torch.Tensor,
    concept_emb: torch.Tensor,
    image_rotary_emb,
    config: Config,
    device: str,
    frame_mask: torch.Tensor | None = None,
    loss_space: str = "velocity",
    x0_target: torch.Tensor | None = None,
    esd_eta: float | None = None,
) -> torch.Tensor:
    """One SFT loss: noise ``x0_input`` at a random timestep, predict velocity, MSE against target.

    Shared by the erase branch (toward an edited fireless latent) and the retention branch
    (toward an unedited preservation latent); see the offline-trainer header for the SNR-shift
    reasoning behind uniform integer timesteps.

    ``x0_target`` defaults to ``x0_input`` — plain reconstruction (noise a latent, predict it
    back). Passing a *different* ``x0_target`` decouples the two: the erase branch noises the
    original fire latent (``x0_input``) but regresses toward the edited fireless latent
    (``x0_target``), turning reconstruction into a fire->fireless denoising redirection.

    ``loss_space`` picks what the MSE compares. ``"velocity"`` matches the predicted velocity
    (the noise-schedule scales the edit down, swamping it at high t). ``"x0"`` recovers the
    predicted clean latent and matches that against the target, so the edit is the full signal at
    every timestep — used by the erase branch when erasure stalls.

    ``frame_mask`` (erase branch only) restricts the MSE to the edited frames: unedited frames
    match the base model's own output, so averaging them in dilutes — and even reinforces — the
    fire behavior we want to remove. ``None`` gives the plain full-tensor mean (retention branch).

    ``esd_eta`` (erase branch only) turns the plain donor target into an ESD-style interpolation:
    instead of regressing fully to the donor, the target becomes
    ``teacher - eta*(teacher - donor)``, where ``teacher`` is the frozen base model's prediction on
    the same noisy input+prompt (computed with the LoRA adapter disabled). ``None`` keeps the plain
    donor target (equivalent to ``eta == 1``). The interpolation is done in whichever space
    ``loss_space`` selects, so ``teacher`` is the base velocity (velocity space) or the base
    predicted-x0 (x0 space), matching ``pred`` and the donor target.
    """
    if x0_target is None:
        x0_target = x0_input

    t = torch.randint(config.timestep_min, config.timestep_max, (x0_input.shape[0],), device=device)
    noise = torch.randn_like(x0_input)
    x_t = scheduler.add_noise(x0_input, noise, t)

    v_pred = transformer(
        hidden_states=x_t.permute(0, 2, 1, 3, 4),  # -> (B, F, C, H, W)
        encoder_hidden_states=concept_emb,
        timestep=t,
        image_rotary_emb=image_rotary_emb,
    ).sample  # (B, F, C, H, W)

    # Base-model (LoRA-disabled) prediction on the same noised input+prompt, used only to build the
    # ESD interpolation target below. No grad: it is a fixed reference the student regresses toward.
    v_pred_teacher = None
    if esd_eta is not None:
        with torch.no_grad(), transformer.disable_adapter():
            v_pred_teacher = transformer(
                hidden_states=x_t.permute(0, 2, 1, 3, 4),  # -> (B, F, C, H, W)
                encoder_hidden_states=concept_emb,
                timestep=t,
                image_rotary_emb=image_rotary_emb,
            ).sample  # (B, F, C, H, W)

    if loss_space == "x0":
        x_t = x_t.permute(0, 2, 1, 3, 4)  # -> (B, F, C, H, W)
        pred = _predict_x0(x_t, v_pred, t, scheduler)
        donor_target = x0_target.permute(0, 2, 1, 3, 4).float()
        teacher_ref = _predict_x0(x_t, v_pred_teacher, t, scheduler) if v_pred_teacher is not None else None
    elif loss_space == "velocity":
        pred = v_pred.float()
        donor_target = scheduler.get_velocity(x0_target, noise, t).permute(0, 2, 1, 3, 4).float()
        teacher_ref = v_pred_teacher.float() if v_pred_teacher is not None else None
    else:
        raise ValueError(f"Unknown loss_space {loss_space!r}; expected 'velocity' or 'x0'.")

    # target = teacher - eta*(teacher - donor) = (1-eta)*teacher + eta*donor; detached so only the
    # student prediction carries gradient. Plain donor target when esd_eta is unset.
    if teacher_ref is None:
        target = donor_target
    else:
        target = (teacher_ref - esd_eta * (teacher_ref - donor_target)).detach()

    if frame_mask is None:
        return F.mse_loss(pred, target)

    se = (pred - target) ** 2
    weighted = se * frame_mask
    return weighted.sum() / frame_mask.expand_as(se).sum().clamp(min=1.0)


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

    retention_metadata: list[dict] = []
    if config.retention_metadata_file is not None:
        with open(config.retention_metadata_file) as f:
            retention_metadata = json.load(f)
        if not retention_metadata:
            raise ValueError(f"No entries in {config.retention_metadata_file}; retention is enabled but empty.")
        if config.retention_exclude is not None:
            kept = [e for e in retention_metadata if e.get("class_name") != config.retention_exclude]
            if len(kept) == len(retention_metadata):
                raise ValueError(
                    f"retention_exclude={config.retention_exclude!r} matched no anchor; "
                    f"{config.retention_metadata_file} has no such 'class_name' (typo, or the wrong dataset)."
                )
            print(f"Retention: dropped {len(retention_metadata) - len(kept)} anchors for {config.retention_exclude!r}")
            retention_metadata = kept
        if config.retention_latents_dir is None:
            raise ValueError("retention_metadata_file is set but retention_latents_dir is not.")
        retention_scaling = float(retention_metadata[0].get("scaling_factor", expected_scaling))
        assert abs(retention_scaling - expected_scaling) < 1e-6, (
            f"Retention latents use scaling_factor {retention_scaling}, model uses {expected_scaling}."
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
    

    if config.gradient_accumulation_steps < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {config.gradient_accumulation_steps}")

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate)
    if config.lr_scheduler == "cosine":
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.steps, eta_min=COSINE_LR_FLOOR_FRACTION * config.learning_rate
        )
    elif config.lr_scheduler == "constant":
        main_sched = None
    else:
        raise ValueError(f"Unknown lr_scheduler {config.lr_scheduler!r}; expected 'constant' or 'cosine'.")

    if config.lr_warmup_steps > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=config.lr_warmup_steps
        )
        lr_sched = (
            torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, main_sched], milestones=[config.lr_warmup_steps]
            )
            if main_sched is not None
            else warmup_sched  # LinearLR holds at end_factor (full LR) once total_iters is exceeded
        )
    else:
        lr_sched = main_sched

    # Cache one T5 embedding per unique prompt (CFG-free; we handle no guidance here).
    # Both the erase and retention targets are keyed by prompt, so cache both prompt sets.
    prompt_emb_cache: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        all_prompts = {entry["prompt"] for entry in metadata} | {entry["prompt"] for entry in retention_metadata}
        for prompt in all_prompts:
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
            "lr_scheduler": config.lr_scheduler,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "steps": config.steps,
            "save_interval": config.save_interval,
            "timestep_min": config.timestep_min,
            "timestep_max": config.timestep_max,
            "num_targets": len(metadata),
            "num_retention_targets": len(retention_metadata),
            "retention_weight": config.retention_weight if retention_metadata else 0.0,
            "nonfire_frame_weight": config.nonfire_frame_weight,
            "erase_loss_space": config.erase_loss_space,
            "erase_input_latent": config.erase_input_latent,
            "erase_esd_eta": config.erase_esd_eta,
            "eval_num_prompts": config.eval_num_prompts,
            "eval_inference_steps": config.eval_inference_steps,
            "global_seed": config.global_seed,
        },
        flush_interval=config.metrics_log_interval,
    )

    retention_enabled = bool(retention_metadata)
    print(
        f"Starting frame_replace SFT over {len(metadata)} edited targets"
        + (f" + {len(retention_metadata)} retention anchors (w={config.retention_weight})" if retention_enabled else "")
        + "..."
    )
    accum = config.gradient_accumulation_steps
    pbar = tqdm(range(config.steps))
    for step in pbar:
        optimizer.zero_grad()
        erase_loss_sum = 0.0
        retain_loss_sum = 0.0
        # Each micro-step draws an independent (target, timestep, noise) sample; gradients are
        # scaled by 1/accum so the single optimizer step sees the micro-batch mean. Losses are
        # backpropped per branch so the two forward graphs never coexist in memory (lower peak
        # than backprop on a summed loss).
        for _ in range(accum):
            # Erase branch: pull the fire prompt toward its edited (fireless) latent.
            erase_entry = random.choice(metadata)
            x0_edited = _load_target_latent(config.latents_dir, erase_entry["latent_path"], device)
            # "original" noises the pre-edit fire latent (x0_input) while still regressing toward
            # the edited fireless latent (x0_target) — a fire->fireless redirection on the states
            # the model actually traverses. "edited" keeps plain reconstruction (x0_input == x0_target).
            if config.erase_input_latent == "original":
                assert "original_latent_path" in erase_entry, (
                    "erase_input_latent='original' needs 'original_latent_path' in the target metadata; "
                    "re-run the precompute with the original-latent-saving version."
                )
                x0_input = _load_target_latent(config.latents_dir, erase_entry["original_latent_path"], device)
            else:
                x0_input = x0_edited
            erase_frame_mask = _fire_frame_mask(erase_entry, config.nonfire_frame_weight, device)
            loss_erase = _sft_velocity_loss(
                transformer, scheduler, x0_input, prompt_emb_cache[erase_entry["prompt"]],
                image_rotary_emb, config, device, frame_mask=erase_frame_mask,
                loss_space=config.erase_loss_space, x0_target=x0_edited,
                esd_eta=config.erase_esd_eta,
            )
            (loss_erase / accum).backward()
            erase_loss_sum += loss_erase.item()

            if retention_enabled:
                # Retention branch: independent sample/timestep anchors a preservation prompt to
                # the base model's unedited latent, keeping erasure from collapsing general quality.
                retain_entry = random.choice(retention_metadata)
                x0_retain = _load_target_latent(config.retention_latents_dir, retain_entry["latent_path"], device)
                loss_retain = _sft_velocity_loss(
                    transformer, scheduler, x0_retain, prompt_emb_cache[retain_entry["prompt"]],
                    image_rotary_emb, config, device,
                )
                (config.retention_weight * loss_retain / accum).backward()
                retain_loss_sum += loss_retain.item()

        optimizer.step()
        current_lr = optimizer.param_groups[0]["lr"]
        if lr_sched is not None:
            lr_sched.step()

        loss_erase_mean = erase_loss_sum / accum
        loss_retain_mean = retain_loss_sum / accum
        loss_total = loss_erase_mean + config.retention_weight * loss_retain_mean
        train_metrics = {"train/loss": loss_total, "train/loss_erase": loss_erase_mean, "train/lr": current_lr}
        if retention_enabled:
            train_metrics["train/loss_retain"] = loss_retain_mean
        recorder.log_train(step, train_metrics)
        if not config.disable_mlflow:
            mlflow.log_metrics(train_metrics, step=step)
        wandb.log(train_metrics, step=step)
        pbar.set_description(
            f"loss={loss_total:.4f}" + (f" (e={loss_erase_mean:.4f} r={loss_retain_mean:.4f})" if retention_enabled else "")
        )

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
                        "concept_detection_rate": s["concept_detection_rate"],
                        "concept_area_score_mean": s["concept_area_score_mean"],
                        "clip_score_mean": s["clip_score_mean"],
                        "colorfulness_mean": s["colorfulness_mean"],
                        "motion_score_mean": s["motion_score_mean"],
                        "dover_technical_mean": s["dover_technical_mean"],
                        "dover_aesthetic_mean": s["dover_aesthetic_mean"],
                    }
                    for set_name, s in eval_metrics.items()
                },
            })

    recorder.close()
    print("frame_replace training complete.")
