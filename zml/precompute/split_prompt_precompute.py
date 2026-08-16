"""Temporal split-prompt generation for building *partial-concept* clips.

Motivation. ``frame_replace`` needs clips where the target concept is present in only *some* frames,
with concept-free "donor" frames elsewhere. Fire gives that for free (it flickers in and out); nudity
and characters do not — the concept is present the whole clip, so there are no donors and the method
has nothing to build a target from. This algorithm *manufactures* the partiality: it steers the first
temporal half of the video with a concept-free prompt (B) and the second half with a concept prompt
(A), then heals the seam with a shared neutral prompt (C), yielding one coherent clip that is
concept-free early and concept-bearing late — donors and concept frames in a single on-distribution
generation.

Mechanism (MultiDiffusion-style, no attention surgery). For the first ``split_step_frac`` of the
denoising schedule, each step runs the transformer twice — once conditioned on prompt A, once on
prompt B — and assembles one velocity where latent frames ``[:split_latent_frame]`` take B's
prediction and ``[split_latent_frame:]`` take A's, then does a single scheduler step on the full
latent (so the DPM-solver state stays coherent). After the split phase, every step conditions on the
shared neutral prompt C (or, with ``tail_prompt_mode="empty"``, on nothing at all), letting the two
halves denoise into a temporally coherent whole while keeping the layout each half already committed
to.

This first milestone just *generates and saves* the four videos per row — A, B, C (each a plain
generation) and the combined split clip — sharing one initial noise per row so they are directly
comparable (and so A/B double as a paired same-seed donor baseline). Fire/nudity detection and the
donor edit that turn the combined clip into a frame_replace target are a later step.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass

import pandas as pd
import torch
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from tqdm import tqdm

from zml.unlearn.frame_replace_ops import (
    EXPECTED_LATENT_SHAPE,
    NUM_LATENT_FRAMES,
    NUM_PIXEL_FRAMES,
    decode_to_bgr_frames,
    write_mp4,
)

DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    # CSV with columns: prompt_a (concept), prompt_b (concept-free), prompt_c (neutral shared), seed.
    csv_path: str
    model_id: str = "THUDM/CogVideoX-5b"
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    num_frames: int = NUM_PIXEL_FRAMES
    height: int = 480
    width: int = 720
    # Latent frame index that splits the two regions. Default ~halfway of the 13 latent frames.
    split_latent_frame: int = 7
    # Which temporal region carries the concept (prompt A); the other carries prompt B (concept-free).
    # "second" -> A on frames [split:]; "first" -> A on [:split]; "random" -> per-clip coin flip.
    # Randomizing across the dataset breaks the "copy the concept-free half onto the other half"
    # positional shortcut the trainer could otherwise learn instead of removing the concept.
    concept_region: str = "second"
    # Per-clip jitter (+/-) on split_latent_frame so the boundary is not always at a fixed position
    # (another positional cue to decorrelate). 0 disables. Resolved with a per-clip seeded RNG.
    split_jitter: int = 0
    # Fraction of the denoising schedule to keep the A/B split before switching to the shared tail
    # conditioning. Early steps set global content (is the concept there?); the tail heals the seam.
    #
    # Measured range of authority (exp099, 5 scenes x {0.5, 0.85}): above ~0.5 this knob is inert for
    # *content*. The same seed at 0.5 and 0.85 gives clips differing by 2-4 grey levels — texture, not
    # subject — and every clip keeps its two-state/collapsed verdict. Content is committed in the first
    # ~20 of 50 steps, so a switch placed after that only refines what is already decided. exp074's
    # finding that 0.2/0.3 wash the concept out is the same fact from the other side: those put the
    # switch *inside* the decisive window, where the tail prompt does win. Do not expect 0.85 vs 1.0 to
    # change anything; the levers that move object yield are prompt framing and (prompt, seed) choice.
    split_step_frac: float = 0.85
    # What conditions the tail (post-split) phase.
    #   "c"     - the shared neutral prompt C, as originally designed.
    #   "empty" - the empty string. With CFG the positive and negative embeddings then coincide, so
    #             the guidance term vanishes and the tail is *pure unconditional* denoising: it
    #             sharpens whatever the split phase committed to without arguing for or against any
    #             content. This matters for object classes, where prompt C is necessarily a
    #             concept-*deleting* prompt ("a wooden workbench in a cluttered garage" for chain saw),
    #             so every C step pushes against the very concept the A-half is supposed to keep.
    #             Nudity does not have this problem — its C keeps the subject and leaves only the
    #             clothed/naked attribute open — which is why the tail never looked harmful there.
    tail_prompt_mode: str = "c"
    # CFG scale applied to the *concept* branch (pred_a) only; None reuses guidance_scale for both.
    #
    # Why the concept side needs its own scale. Both branches are predicted over the whole latent and
    # only the prediction is spliced (see generate_split_clip), so pred_a is evaluated in a context
    # whose other region is converging on prompt B. CogVideoX's temporal-coherence prior then argues
    # that the clip is one scene, and the concept region gets pulled toward the substitute. exp117's
    # whole-clip diagnostics show this is now the dominant loss and that it fails *binary*: rows that
    # survive keep 112% of the concept confidence a plain prompt-A generation reaches at the same
    # seed, rows that fail keep 6%. Raising only this branch's guidance strengthens the concept
    # region's own conditioning against that pull, and costs nothing — it is a scalar on predictions
    # that are already computed.
    concept_guidance_scale: float | None = None
    # Where the two prompts are combined during the split phase.
    #
    #   "prediction" (default, every dataset up to exp122) - one latent, two predictions per step,
    #       and the *prediction* is spliced. Cheap and coherent, but pred_a is evaluated on a latent
    #       whose other region is converging on prompt B, so the concept region is under constant
    #       pull from the substitute (see concept_guidance_scale above).
    #   "trajectory" - two latents from the same initial noise, each denoised under its own prompt,
    #       spliced ONCE at split_step and healed by the tail from there. The concept region is then
    #       denoised in a pure-A context and the safe region never sees prompt A at all.
    #
    # Same cost either way: two transformer calls per split step in both modes.
    #
    # Why this exists: exp120 confirmed the pull is real and rejected CFG as the cure. Raising
    # concept_guidance to 9 brought the concept back in 7 of 12 suppressed rows, but 5 of those 7 then
    # had the concept in BOTH halves — the branch's own context is the coupling, so the fix has to be
    # in the context, not in the guidance scale. Cost to weigh: coherence across the seam comes from
    # the shared initial noise (exp076 found the cut is hard at every split_step_frac, including with
    # zero heal steps), but two independent trajectories share less than two spliced predictions do.
    split_mode: str = "prediction"
    output_dir: str = "."
    videos_subdir: str = "videos"
    # Save the combined + A + B clean latents (for later donor-edit / paired-baseline dataset use).
    save_latents: bool = True
    # Skip the plain A/B/C generations and only produce the "combined" split clip. A/B/C don't depend
    # on split_latent_frame/split_step_frac, so a hyperparameter sweep over those (many grid jobs, same
    # rows) would otherwise regenerate three identical clips per row in every job for no reason.
    skip_plain_abc: bool = False


def resolve_split(config: Config, rng: random.Random) -> tuple[int, str]:
    """Per-clip (split_latent_frame, concept_region), applying jitter and random-side if configured."""
    sf = config.split_latent_frame
    if config.split_jitter:
        sf += rng.randint(-config.split_jitter, config.split_jitter)
    sf = max(1, min(NUM_LATENT_FRAMES - 1, sf))
    region = rng.choice(["first", "second"]) if config.concept_region == "random" else config.concept_region
    if region not in ("first", "second"):
        raise ValueError(f"concept_region must be 'first'|'second'|'random', got {config.concept_region!r}")
    return sf, region


SPLIT_MODES = ("prediction", "trajectory")


def validated_split_mode(mode: str) -> str:
    """``Config.split_mode``, rejected early rather than silently falling back to the default."""
    if mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {mode!r}")
    return mode


TAIL_PROMPT_MODES = ("c", "empty")


def tail_prompt(prompt_c: str, mode: str) -> str:
    """The prompt conditioning the post-split (heal) phase, per ``Config.tail_prompt_mode``."""
    if mode not in TAIL_PROMPT_MODES:
        raise ValueError(f"tail_prompt_mode must be one of {TAIL_PROMPT_MODES}, got {mode!r}")
    return "" if mode == "empty" else prompt_c


def _cfg_embeds(pipe, prompt: str, do_cfg: bool):
    """Encode ``prompt`` and return the [neg, pos] CFG batch (or just pos when CFG is off)."""
    pos, neg = pipe.encode_prompt(prompt, None, do_cfg, device=pipe._execution_device)
    return torch.cat([neg, pos], dim=0) if do_cfg else pos


def _predict(pipe, latents, embeds, t, rope, guidance_scale: float, do_cfg: bool) -> torch.Tensor:
    """One CFG-combined velocity prediction for ``latents`` conditioned on ``embeds`` at timestep ``t``."""
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    timestep = t.expand(latent_model_input.shape[0])
    pred = pipe.transformer(
        hidden_states=latent_model_input,
        encoder_hidden_states=embeds,
        timestep=timestep,
        image_rotary_emb=rope,
        return_dict=False,
    )[0].float()
    if do_cfg:
        uncond, text = pred.chunk(2)
        pred = uncond + guidance_scale * (text - uncond)
    return pred


@torch.no_grad()
def generate_plain_clip(pipe, prompt: str, seed: int, config: Config) -> torch.Tensor:
    """Vanilla generation -> clean latent in (B, C, F, H, W). Fresh generator so init noise == seed.

    Public (promoted from ``_generate_plain``) because ``frame_replace_split_precompute.py`` reuses
    it directly for the whole-clip target variant (``emit_whole_clip_target``): prompt A's own plain
    clip as the "original" and prompt B's same-seed plain clip as the donor, generated the same way
    the paired A/B baseline here always has been.
    """
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    out = pipe(
        prompt=prompt,
        num_frames=config.num_frames,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
        output_type="latent",
    )
    return out.frames.permute(0, 2, 1, 3, 4).contiguous()  # (B, F, C, H, W) -> (B, C, F, H, W)


# Private alias kept for anything still importing the old name.
_generate_plain = generate_plain_clip


def _splice(concept_side: torch.Tensor, safe_side: torch.Tensor, sf: int, concept_region: str) -> torch.Tensor:
    """``safe_side`` everywhere, with the concept region taken from ``concept_side``.

    Both tensors are (B, F, C, H, W) with frames on dim 1, and both modes splice on the same frame
    index — the difference is only whether the tensors are predictions or latents.
    """
    out = safe_side.clone()
    if concept_region == "first":
        out[:, :sf] = concept_side[:, :sf]
    else:  # "second"
        out[:, sf:] = concept_side[:, sf:]
    return out


def _scheduler_step(pipe, noise_pred, latents, t, t_back, old_pred, extra_step_kwargs):
    """One denoising step -> ``(latents, old_pred)``, mirroring the pipeline.

    DDIM takes ``(pred, t, sample)``; DPM-solver++ is a multistep whose state is the *caller's*
    ``old_pred``. The scheduler object keeps no step counter of its own — it derives the previous
    timestep from the one passed in — which is what lets two trajectories share one scheduler.
    """
    if not isinstance(pipe.scheduler, CogVideoXDPMScheduler):
        return pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0], None
    return pipe.scheduler.step(
        noise_pred, old_pred, t, t_back, latents, **extra_step_kwargs, return_dict=False,
    )


@torch.no_grad()
def generate_split_clip(
    pipe, prompt_a: str, prompt_b: str, prompt_c: str, seed: int, config: Config,
    split_latent_frame: int, concept_region: str,
) -> torch.Tensor:
    """Temporal split-prompt generation -> clean latent in (B, C, F, H, W).

    ``concept_region`` selects which side gets the concept prompt A: "second" -> frames
    ``[split:]``; "first" -> ``[:split]``. Shares its initial noise with ``generate_plain_clip`` for the
    same seed, so the combined clip is comparable to the plain A/B/C clips. Reused by the dataset
    builder (``frame_replace_split_precompute``).

    ``config.split_mode`` chooses *where* the two prompts are combined during the split phase; both
    modes cost the same two transformer calls per step. See ``Config.split_mode``.
    """
    device = pipe._execution_device
    concept_guidance = config.concept_guidance_scale or config.guidance_scale
    do_cfg = max(config.guidance_scale, concept_guidance) > 1.0
    trajectory = validated_split_mode(config.split_mode) == "trajectory"
    emb_a = _cfg_embeds(pipe, prompt_a, do_cfg)
    emb_b = _cfg_embeds(pipe, prompt_b, do_cfg)
    emb_tail = _cfg_embeds(pipe, tail_prompt(prompt_c, config.tail_prompt_mode), do_cfg)

    pipe.scheduler.set_timesteps(config.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    num_steps = len(timesteps)
    split_step = int(round(config.split_step_frac * num_steps))
    sf = split_latent_frame

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    latents = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels, config.num_frames,
        config.height, config.width, emb_a.dtype, device, generator, None,
    )  # (B, F, C, H, W), frames on dim 1
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)
    rope = (
        pipe._prepare_rotary_positional_embeddings(config.height, config.width, latents.size(1), device)
        if pipe.transformer.config.use_rotary_positional_embeddings else None
    )

    # In trajectory mode `latents` is the A-conditioned trajectory and `latents_b` the B-conditioned
    # one, both from the same initial noise; each carries its own multistep state.
    latents_b = latents.clone() if trajectory else None
    old_pred = old_pred_b = None
    for i, t in enumerate(timesteps):
        t_back = timesteps[i - 1] if i > 0 else None
        if trajectory and i == split_step:  # happens once; after it the tail runs on one latent
            latents = _splice(latents, latents_b, sf, concept_region)
        if i < split_step:
            pred_a = _predict(pipe, latents, emb_a, t, rope, concept_guidance, do_cfg)
            if trajectory:
                pred_b = _predict(pipe, latents_b, emb_b, t, rope, config.guidance_scale, do_cfg)
                latents_b, old_pred_b = _scheduler_step(
                    pipe, pred_b, latents_b, t, t_back, old_pred_b, extra_step_kwargs)
                latents_b = latents_b.to(emb_a.dtype)
                noise_pred = pred_a  # the whole A trajectory advances on A; only its concept half is kept
            else:
                pred_b = _predict(pipe, latents, emb_b, t, rope, config.guidance_scale, do_cfg)
                noise_pred = _splice(pred_a, pred_b, sf, concept_region)
        else:
            noise_pred = _predict(pipe, latents, emb_tail, t, rope, config.guidance_scale, do_cfg)  # heal seam
        latents, old_pred = _scheduler_step(pipe, noise_pred, latents, t, t_back, old_pred, extra_step_kwargs)
        latents = latents.to(emb_a.dtype)

    if trajectory and split_step >= num_steps:  # split_step_frac 1.0: no tail step reached the merge
        latents = _splice(latents, latents_b, sf, concept_region)

    return latents.permute(0, 2, 1, 3, 4).contiguous()  # (B, F, C, H, W) -> (B, C, F, H, W)


def _save(pipe, z_bcfhw: torch.Tensor, stem: str, tag: str, videos_dir: str, latents_dir, save_latents: bool) -> str:
    assert z_bcfhw.shape == EXPECTED_LATENT_SHAPE, f"unexpected latent shape {z_bcfhw.shape} for {stem}_{tag}"
    video_path = os.path.join(videos_dir, f"{stem}_{tag}.mp4")
    write_mp4(decode_to_bgr_frames(pipe, z_bcfhw), video_path)
    if save_latents and latents_dir is not None:
        torch.save(z_bcfhw.cpu(), os.path.join(latents_dir, f"{stem}_{tag}.pt"))
    return os.path.relpath(video_path, os.path.dirname(videos_dir))


def main(config: Config) -> None:
    videos_dir = os.path.join(config.output_dir, config.videos_subdir)
    os.makedirs(videos_dir, exist_ok=True)
    latents_dir = os.path.join(config.output_dir, "latents") if config.save_latents else None
    if latents_dir is not None:
        os.makedirs(latents_dir, exist_ok=True)

    pipe = CogVideoXPipeline.from_pretrained(config.model_id, torch_dtype=DTYPE).to(DEVICE)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    assert pipe.scheduler.config.prediction_type == "v_prediction", (
        f"Expected v_prediction scheduler, got {pipe.scheduler.config.prediction_type!r}"
    )
    assert 0 < config.split_latent_frame < NUM_LATENT_FRAMES, (
        f"split_latent_frame must be in (0, {NUM_LATENT_FRAMES}), got {config.split_latent_frame}"
    )

    df = pd.read_csv(config.csv_path)
    for col in ("prompt_a", "prompt_b", "prompt_c", "seed"):
        if col not in df.columns:
            raise ValueError(f"{config.csv_path} missing required column '{col}'.")

    metadata: list[dict] = []
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            seed = int(row["seed"])
            stem = f"p{idx}_s{seed}"
            sf, region = resolve_split(config, random.Random(seed))
            clips = {
                "combined": generate_split_clip(
                    pipe, row["prompt_a"], row["prompt_b"], row["prompt_c"], seed, config, sf, region),
            }
            if not config.skip_plain_abc:
                clips["A"] = generate_plain_clip(pipe, row["prompt_a"], seed, config)
                clips["B"] = generate_plain_clip(pipe, row["prompt_b"], seed, config)
                clips["C"] = generate_plain_clip(pipe, row["prompt_c"], seed, config)
            paths = {tag: _save(pipe, z, stem, tag, videos_dir, latents_dir, config.save_latents)
                     for tag, z in clips.items()}
            metadata.append({
                "stem": stem, "seed": seed,
                "prompt_a": row["prompt_a"], "prompt_b": row["prompt_b"], "prompt_c": row["prompt_c"],
                "split_latent_frame": sf, "concept_region": region, "split_step_frac": config.split_step_frac,
                "tail_prompt_mode": config.tail_prompt_mode, "split_mode": config.split_mode,
                "videos": paths,
            })
            with open(os.path.join(config.output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

    print(f"split-prompt precompute done: {len(metadata)} rows x 4 clips -> {videos_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="CSV with prompt_a/prompt_b/prompt_c/seed")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    main(Config(csv_path=args.csv_path, output_dir=args.output_dir))
