"""Fully-online ``frame_replace`` unlearning.

Unlike the offline variant (``unlearn_frame_replace.py``), which fine-tunes toward edited target
latents precomputed once by the *base* model, this trainer generates the targets *online* with the
current *student* (LoRA-applied) model:

    generate a clip -> decode -> detect fire per frame -> replace fire latent frames with the
    nearest fire-free donor -> SFT (v-prediction MSE) toward the edited latent.

This makes the method a self-curriculum: as the LoRA suppresses fire, the student's own clips
contain less fire, the edit shrinks toward identity, and the loss tapers naturally.

Generation (a full diffusion rollout + VAE decode + YOLO) is ~15-20x a single SFT step, so we
amortize it behind a replay buffer: a clip is generated only every ``regen_interval`` SFT steps,
pushed into a fixed-capacity buffer, and every SFT step samples a target from that buffer.
"""

import json
import os
import random
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
import wandb
import torch
import torch.nn.functional as F
from diffusers import CogVideoXPipeline
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from zml.eval.check_for_fire import VideoFireDetector
from zml.unlearn.eval import EvalPrompt, evaluate
from zml.unlearn.frame_replace_ops import (
    EXPECTED_LATENT_SHAPE,
    NUM_LATENT_FRAMES,
    NUM_PIXEL_FRAMES,
    VIDEO_FPS,
    build_latent_fire_mask,
    decode_to_bgr_frames,
    edit_latent,
    write_mp4,
)
from zml.unlearn.metrics_log import MetricsRecorder
from zml.utils import set_seed

LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
DTYPE = torch.bfloat16


@dataclass
class Config:
    model_id: str
    train_prompts: str  # CSV with 'prompt' + 'seed' columns; clips are generated online from these trusted pairs
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
    # Online generation / replay buffer
    buffer_size: int = 8  # how many recent edited targets to keep
    regen_interval: int = 40  # SFT steps between generating one fresh target
    generate_inference_steps: int = 50  # denoise steps for online generation
    guidance_scale: float = 6.0
    frame_fire_threshold: float = 0.5  # per-frame fire confidence above which a frame counts as fire
    min_nofire_frames: int = 2  # need at least this many fire-free latent frames to build donors
    max_generation_attempts: int = 8  # retries to find a fire clip before giving up this refresh
    # SFT timestep sampling (see offline trainer for the SNR-shift rationale)
    timestep_min: int = 0
    timestep_max: int = 1000
    num_frames: int = NUM_PIXEL_FRAMES  # generation geometry (pixel frames)
    height: int = 480  # pixel height — used for rotary embeddings
    width: int = 720  # pixel width
    # Saving the student's generated clips (and their edited targets) lets us watch the
    # self-curriculum: fire should shrink across the run. Generation already decodes both clips for
    # fire detection, so writing them is nearly free; the pool is small so we save all by default.
    save_generated_videos: bool = True
    save_videos_every_n_gens: int = 1  # save 1 of every N accepted generations (1 = save all)
    video_fps: int = VIDEO_FPS
    videos_subdir: str = "train_videos"
    global_seed: int | None = None
    disable_mlflow: bool = False
    metrics_log_interval: int = 50


@dataclass
class TrainPrompt:
    """A trusted (prompt, seed) pair that reliably renders partial fire (see seed policy)."""
    prompt: str
    seed: int


class PromptQueue:
    """Draws trusted (prompt, seed) pairs without replacement, reshuffling once a pass is empty.

    The queue's state persists *across* buffer refreshes, so every pair is generated once per pass
    before any repeats. This gives even coverage of the (small) train pool — random sampling would
    over-hit some pairs and starve others, and with a LoRA on a tiny pool that risks overfitting to
    a handful of clips. Within a single refresh it also guarantees distinct pairs: re-generating the
    same pair would be pointless, since its seed and the (frozen) student weights are fixed there, so
    the result is identical.
    """

    def __init__(self, prompts: list[TrainPrompt]) -> None:
        self._prompts = prompts
        self._queue: list[TrainPrompt] = []

    @property
    def size(self) -> int:
        return len(self._prompts)

    def next(self) -> TrainPrompt:
        if not self._queue:
            self._queue = random.sample(self._prompts, len(self._prompts))
        return self._queue.pop()


@dataclass
class PromptAttempt:
    """Outcome of generating from one (prompt, seed) pair during a buffer refresh."""
    prompt: str
    seed: int
    max_confidence: float  # peak per-frame fire confidence on the generated (pre-edit) clip
    num_nofire_frames: int  # fire-free latent frames available as donors
    accepted: bool


@dataclass
class GenerationResult:
    """Result of one buffer refresh: an optional target plus the per-pair attempt log."""
    target: "Target | None"  # None if no usable fire clip was produced this refresh
    attempts: list[PromptAttempt]
    # Decoded clips for the accepted target (None if nothing accepted), reused for video saving.
    original_frames: list[np.ndarray] | None = None
    edited_frames: list[np.ndarray] | None = None


@dataclass
class Target:
    """One supervised SFT target generated online."""
    prompt_emb: torch.Tensor  # cached conditional T5 embedding for the prompt
    x0_edited: torch.Tensor  # (1, C, F, H, W) scaled latent with fire frames replaced
    edited_max_confidence: float  # peak fire confidence on the edited clip (curriculum signal)


class TargetBuffer:
    """Fixed-capacity ring of online targets; SFT samples uniformly at random from it.

    Latents live on CPU (one ~kept-alive clip per slot would otherwise pin VRAM) and are moved to
    the training device on ``sample``; the small T5 embedding is kept on-device.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._items: list[Target] = []

    def add(self, target: Target) -> None:
        self._items.append(target)
        if len(self._items) > self._capacity:
            self._items.pop(0)

    def sample(self) -> Target:
        return random.choice(self._items)

    def __len__(self) -> int:
        return len(self._items)


def _load_eval_prompts(path: str) -> list[EvalPrompt]:
    df = pd.read_csv(path)
    return [EvalPrompt(prompt=row["prompt"], seed=int(row["seed"])) for _, row in df.iterrows()]


def _load_train_prompts(path: str) -> list[TrainPrompt]:
    df = pd.read_csv(path)
    return [TrainPrompt(prompt=row["prompt"], seed=int(row["seed"])) for _, row in df.iterrows()]


def _embed_prompt(
    pipe: CogVideoXPipeline, prompt: str, cache: dict[str, torch.Tensor], device: str
) -> torch.Tensor:
    """Conditional (CFG-free) T5 embedding used by the SFT step, cached per unique prompt."""
    if prompt not in cache:
        with torch.no_grad():
            embeds, _ = pipe.encode_prompt(prompt=prompt, do_classifier_free_guidance=False)
        cache[prompt] = embeds.to(device, dtype=DTYPE)
    return cache[prompt]


def generate_one_target(
    pipe: CogVideoXPipeline,
    transformer,
    detector: VideoFireDetector,
    config: Config,
    prompt_queue: PromptQueue,
    prompt_emb_cache: dict[str, torch.Tensor],
    device: str,
) -> GenerationResult:
    """Generate, fire-check and edit one clip with the current student.

    Each clip is generated from a trusted ``(prompt, seed)`` pair drawn from ``prompt_queue`` using
    that pair's *attached* seed (not the global seed) — see the seed policy in CLAUDE.md. The
    student still evolves, so the same pair yields progressively less fire over training (the
    self-curriculum).

    Tries distinct pairs (no repeats within a refresh) until one passes the skip rules (has fire and
    enough donor frames) or we exhaust ``max_generation_attempts`` / the whole pool. Returns a
    ``GenerationResult`` whose ``target`` is ``None`` if no usable fire clip was found — the expected,
    healthy outcome once the student stops producing fire. Every pair tried is recorded in
    ``attempts`` for the per-pair outcome log.
    """
    was_training = transformer.training
    transformer.eval()
    attempts: list[PromptAttempt] = []
    tried: set[tuple[str, int]] = set()
    try:
        with torch.no_grad():
            while len(attempts) < config.max_generation_attempts and len(tried) < prompt_queue.size:
                tp = prompt_queue.next()
                key = (tp.prompt, tp.seed)
                if key in tried:  # queue wrapped mid-refresh onto a pair we already tried
                    continue
                tried.add(key)

                generator = torch.Generator(device=device).manual_seed(tp.seed)
                out = pipe(
                    prompt=tp.prompt,
                    num_frames=config.num_frames,
                    num_inference_steps=config.generate_inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=generator,
                    output_type="latent",
                )
                # output_type="latent" returns the scaled clean latent in (B, F, C, H, W) layout.
                z_bcfhw = out.frames.permute(0, 2, 1, 3, 4).contiguous()  # -> (B, C, F, H, W)
                assert z_bcfhw.shape == EXPECTED_LATENT_SHAPE, f"unexpected latent shape {z_bcfhw.shape}"

                frames = decode_to_bgr_frames(pipe, z_bcfhw)
                confidences = detector.frame_fire_confidences(frames)
                fire_pixel = [c >= config.frame_fire_threshold for c in confidences]
                fire_latent = build_latent_fire_mask(fire_pixel)
                nofire = [i for i in range(NUM_LATENT_FRAMES) if not fire_latent[i]]

                accepted = any(fire_latent) and len(nofire) >= config.min_nofire_frames
                attempts.append(PromptAttempt(
                    prompt=tp.prompt,
                    seed=tp.seed,
                    max_confidence=max(confidences, default=0.0),
                    num_nofire_frames=len(nofire),
                    accepted=accepted,
                ))
                if not accepted:
                    continue

                x0_edited, _ = edit_latent(z_bcfhw, fire_latent)
                edited_frames = decode_to_bgr_frames(pipe, x0_edited)
                edited_conf = detector.frame_fire_confidences(edited_frames)
                target = Target(
                    prompt_emb=_embed_prompt(pipe, tp.prompt, prompt_emb_cache, device),
                    x0_edited=x0_edited.cpu(),
                    edited_max_confidence=max(edited_conf, default=0.0),
                )
                return GenerationResult(
                    target=target, attempts=attempts,
                    original_frames=frames, edited_frames=edited_frames,
                )
        return GenerationResult(target=None, attempts=attempts)
    finally:
        if was_training:
            transformer.train()


def _append_generation_log(log_path: str, record: dict) -> None:
    """Append one refresh's per-pair outcomes to the generation log (JSONL, one object per refresh)."""
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _save_train_videos(
    result: GenerationResult, videos_dir: str, output_dir: str, step: int, gen_index: int, fps: int
) -> dict[str, str]:
    """Write the accepted clip's original + edited MP4s; return their paths relative to output_dir."""
    seed = result.attempts[-1].seed  # the accepted pair is always the last attempt
    stem = f"step{step:05d}_g{gen_index:03d}_s{seed}"
    paths = {
        "original": os.path.join(videos_dir, f"{stem}_original.mp4"),
        "edited": os.path.join(videos_dir, f"{stem}_edited.mp4"),
    }
    write_mp4(result.original_frames, paths["original"], fps)
    write_mp4(result.edited_frames, paths["edited"], fps)
    return {key: os.path.relpath(path, output_dir) for key, path in paths.items()}


def main(config: Config) -> None:
    if config.global_seed is not None:
        set_seed(config.global_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    control_concept = _load_eval_prompts(config.control_concept_prompts)
    control_related = _load_eval_prompts(config.control_related_prompts)
    control_unrelated = _load_eval_prompts(config.control_unrelated_prompts)

    train_prompts = _load_train_prompts(config.train_prompts)
    if not train_prompts:
        raise ValueError(f"No prompts in {config.train_prompts}.")

    pipe = CogVideoXPipeline.from_pretrained(config.model_id, torch_dtype=DTYPE).to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    scheduler = pipe.scheduler
    assert scheduler.config.prediction_type == "v_prediction", (
        f"Expected v_prediction scheduler, got {scheduler.config.prediction_type!r}"
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
    pipe.transformer = transformer  # generation uses the LoRA-applied student

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate)
    detector = VideoFireDetector(video_dir=config.output_dir)
    prompt_emb_cache: dict[str, torch.Tensor] = {}
    prompt_queue = PromptQueue(train_prompts)

    # Rotary embeddings depend only on the fixed latent geometry, so build them once. The
    # transformer does NOT compute these internally — generation applies RoPE, so the SFT step
    # must too or the LoRA learns to correct a mismatched positional regime.
    image_rotary_emb = (
        pipe._prepare_rotary_positional_embeddings(config.height, config.width, NUM_LATENT_FRAMES, device)
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )

    recorder = MetricsRecorder(
        output_dir=config.output_dir,
        run_name=os.path.basename(config.output_dir.rstrip("/")) or "frame_replace_online",
        config={
            "method": "frame_replace_online",
            "model_id": config.model_id,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "learning_rate": config.learning_rate,
            "steps": config.steps,
            "save_interval": config.save_interval,
            "buffer_size": config.buffer_size,
            "regen_interval": config.regen_interval,
            "generate_inference_steps": config.generate_inference_steps,
            "guidance_scale": config.guidance_scale,
            "timestep_min": config.timestep_min,
            "timestep_max": config.timestep_max,
            "eval_num_prompts": config.eval_num_prompts,
            "eval_inference_steps": config.eval_inference_steps,
            "global_seed": config.global_seed,
        },
        flush_interval=config.metrics_log_interval,
    )

    videos_dir = os.path.join(config.output_dir, config.videos_subdir)
    if config.save_generated_videos:
        os.makedirs(videos_dir, exist_ok=True)
    generation_log_path = os.path.join(config.output_dir, "generation_log.jsonl")
    num_generations = 0  # accepted generations so far; drives save_videos_every_n_gens

    def refresh_buffer(buffer: TargetBuffer) -> None:
        nonlocal num_generations
        result = generate_one_target(
            pipe, transformer, detector, config, prompt_queue, prompt_emb_cache, device
        )
        accepted = result.target is not None
        if accepted:
            buffer.add(result.target)

        saved_videos = None
        if accepted and config.save_generated_videos and num_generations % config.save_videos_every_n_gens == 0:
            saved_videos = _save_train_videos(
                result, videos_dir, config.output_dir, step, num_generations, config.video_fps
            )
        if accepted:
            num_generations += 1

        _append_generation_log(generation_log_path, {
            "step": step,
            "accepted": accepted,
            "num_attempts": len(result.attempts),
            "edited_max_confidence": result.target.edited_max_confidence if accepted else None,
            "attempts": [
                {"prompt": a.prompt, "seed": a.seed, "max_confidence": round(a.max_confidence, 4),
                 "num_nofire_frames": a.num_nofire_frames, "accepted": a.accepted}
                for a in result.attempts
            ],
            **({"videos": saved_videos} if saved_videos else {}),
        })

        recorder.log_train(step, {
            "train/gen_skips": float(len(result.attempts) - (1 if accepted else 0)),
            "train/gen_attempts": float(len(result.attempts)),
            "train/edited_max_confidence": result.target.edited_max_confidence if accepted else 0.0,
        })

    buffer = TargetBuffer(config.buffer_size)
    print(f"Warming up replay buffer to {config.buffer_size} targets...")
    step = 0  # referenced by refresh_buffer's recorder logging during warm-up
    while len(buffer) < config.buffer_size:
        prev_len = len(buffer)
        refresh_buffer(buffer)
        if len(buffer) == prev_len:  # student produced no fire even during warm-up
            raise RuntimeError(
                "Could not generate any partial fire clip during warm-up; check prompts / detector."
            )

    print(f"Starting online frame_replace SFT over {config.steps} steps...")
    pbar = tqdm(range(config.steps))
    for step in pbar:
        if step > 0 and step % config.regen_interval == 0:
            refresh_buffer(buffer)

        target = buffer.sample()
        x0 = target.x0_edited.to(device, dtype=DTYPE)
        assert x0.shape == EXPECTED_LATENT_SHAPE, f"unexpected target shape {x0.shape}"
        concept_emb = target.prompt_emb

        t = torch.randint(config.timestep_min, config.timestep_max, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        x_t = scheduler.add_noise(x0, noise, t)
        v_target = scheduler.get_velocity(x0, noise, t)

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
    print("online frame_replace training complete.")
