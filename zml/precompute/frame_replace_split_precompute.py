"""Build frame_replace targets from split-prompt (manufactured partial-concept) clips.

This is the concept-agnostic counterpart of ``frame_replace_precompute``. Fire is naturally partial
(it flickers in and out) and unpredictable, so that script generates from a single prompt and must
*detect* which frames happen to contain fire. Here the partiality is manufactured by the split-prompt
sampler (``generate_split_clip``): one temporal region is conditioned on concept prompt A, the other
on concept-free prompt B (healed by neutral prompt C), and *we choose the split point ourselves*
(``split_latent_frame`` + ``concept_region``). That means the concept mask is known by construction —
frames ``[sf:]`` or ``[:sf]`` depending on ``concept_region`` — so, unlike the fire builder, no
detector is needed to find it. (An earlier version of this script did rederive the mask via NudeNet
per-frame confidences; that made dataset yield hostage to the detector's known unreliability —
flickering mid-clip on scenes that never changed, near-total misses on close-up crops, and
over-triggering on multi-person scenes where one person's clothed frames still scored "concept
present." See exp078's notes.md, 2026-08-04.) The detector still runs, but purely to log
``frame_confidences`` for human review — it no longer gates what gets kept.

The concept block also always touches a clip edge (one side of the split), which means
``frame_replace_ops.edit_latent``'s two-sided interpolation never actually engages here — it
always falls back to copying the single safe frame nearest the boundary across the whole block, a
known motion-suppression risk (see that function's docstring, exp055). ``edit_latent_reflected``
below replaces that with a reflected/bouncing fill of the safe segment's motion instead, so the
edited region has real (if mirrored) motion rather than a frozen frame, still sourced only from
frames outside ``boundary_margin`` of the boundary.

Which detector runs is chosen by ``concept`` (+ ``concept_target``) through
``zml/benchmarks/registry.py``, so a new concept costs a prompt CSV and a detector, not a fork of
this file.

Two de-biasing knobs (see ``split_prompt_precompute.Config``) matter for the resulting dataset:
``concept_region`` (mix "first"/"second"/"random") and ``split_jitter`` decorrelate concept *position*
from the edit, so the trainer must learn to remove the concept rather than the positional shortcut
"copy the concept-free half onto the other half".

The training prompt for each target is the plain concept prompt A (what we actually want to erase at
inference), not the split construction.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass

import pandas as pd
import torch
from diffusers import CogVideoXPipeline
from tqdm import tqdm

from zml.benchmarks.registry import build_detector
from zml.precompute.split_prompt_precompute import generate_plain_clip, generate_split_clip, resolve_split
from zml.precompute.split_prompt_precompute import Config as SplitConfig
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
    # Split-sampler knobs (see split_prompt_precompute.Config).
    split_latent_frame: int = 7
    concept_region: str = "random"  # mix sides across the dataset to break the positional shortcut
    split_jitter: int = 2
    split_step_frac: float = 0.85
    tail_prompt_mode: str = "c"  # "c" | "empty"; see split_prompt_precompute.Config
    concept_guidance_scale: float | None = None  # CFG on the concept branch only; see split_prompt_precompute.Config
    split_mode: str = "prediction"  # "prediction" | "trajectory"; see split_prompt_precompute.Config
    # The concept latent mask is derived directly from (split_latent_frame, concept_region) — see
    # module docstring. The detector still runs to log per-frame confidences for human review, but
    # frame_concept_threshold no longer gates keep/skip.
    # Per-frame detector score above which a frame counts as containing the concept, for logging
    # only. The scale is detector-specific (NudeNet detection score, ResNet-50 class probability),
    # so it must be recalibrated whenever `concept` changes.
    frame_concept_threshold: float = 0.3
    frame_nudity_threshold: float | None = None  # deprecated alias, kept so exp061's config still loads
    min_donor_frames: int = 2  # skip clips whose known concept-free side has fewer latent frames
    # The heal phase (after split_step_frac) jointly attends over the whole latent conditioned on
    # prompt C, so frames right next to the boundary can carry some bleed from the other side even
    # though the split phase's conditioning was cleanly separated. Because split-prompt's concept
    # block always touches a clip edge, edit_latent's donor is always the single nearest safe frame
    # copied across the whole block (never a two-sided interpolation) — so that one frame's
    # cleanliness matters a lot. boundary_margin excludes this many latent frames closest to the
    # boundary from being used as that donor, pulling it from further inside the safe region.
    boundary_margin: int = 2
    concept: str = "nudity"  # selects the detector, see zml/benchmarks/registry.py
    concept_target: str | None = None  # target within a concept family, e.g. "chain saw" for "object"
    output_dir: str = "."
    save_videos: bool = True
    videos_subdir: str = "videos"
    # Also build a second target variant per kept row: prompt A's own plain clip as "original" and
    # prompt B's same-seed plain clip as the "edited"/donor target — a whole-clip swap rather than a
    # frame-local edit. Costs two extra plain generations per row (~2.3x the per-row generation cost
    # of the split target alone), so off by default. Motivating case: identity is present in every
    # frame and maximally salient (docs/face_identity.md), so a temporally-spliced target risks a
    # visible mid-clip face-swap seam or a heal phase that washes the identity back out — the
    # whole-clip variant is a same-pass hedge, not a separate dataset build, for exactly that
    # failure mode. `unlearn_frame_replace.Config.target_variant` selects which a training run
    # consumes ("split" | "wholeclip"); nonfire_frame_weight is inert under "wholeclip" since every
    # frame is a concept frame by construction.
    emit_whole_clip_target: bool = False

    def __post_init__(self) -> None:
        if self.frame_nudity_threshold is not None:
            self.frame_concept_threshold = self.frame_nudity_threshold

    def split_config(self) -> SplitConfig:
        """Adapt to the split sampler's Config (shared generation + split fields)."""
        return SplitConfig(
            csv_path=self.csv_path, model_id=self.model_id, num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale, num_frames=self.num_frames, height=self.height,
            width=self.width, split_latent_frame=self.split_latent_frame, concept_region=self.concept_region,
            split_jitter=self.split_jitter, split_step_frac=self.split_step_frac, output_dir=self.output_dir,
            tail_prompt_mode=self.tail_prompt_mode, concept_guidance_scale=self.concept_guidance_scale,
            split_mode=self.split_mode,
        )


def build_edit_masks(
    split_latent_frame: int, region: str, boundary_margin: int
) -> tuple[list[bool], list[bool]]:
    """``(concept_latent_mask, edit_mask)`` for a clip split at ``split_latent_frame``.

    The concept mask is known by *construction* — split-prompt chooses which temporal region is
    conditioned on the concept prompt — so it is derived here rather than from a detector, which is
    what the detector-driven version got wrong (it cost exp078 half its yield).

    ``edit_mask`` is the concept block widened by ``boundary_margin`` frames, and is what actually
    gets replaced. The concept block always touches a clip edge in this construction, so donors come
    from one side only; the margin pushes the nearest donor away from the seam, where the heal
    phase's joint cross-attention could have bled concept content across. ``concept_latent_mask``
    stays the true, unwidened mask and is what training uses to weight the erase loss.

    Shared by the builder and by ``tools/reedit_frame_replace_dataset.py`` so a re-edit of an
    existing dataset cannot drift from how it would be built today.
    """
    if region == "second":
        concept = [i >= split_latent_frame for i in range(NUM_LATENT_FRAMES)]
        donor_boundary = max(0, split_latent_frame - boundary_margin)
        edit = [i >= donor_boundary for i in range(NUM_LATENT_FRAMES)]
    else:
        concept = [i < split_latent_frame for i in range(NUM_LATENT_FRAMES)]
        donor_boundary = min(NUM_LATENT_FRAMES, split_latent_frame + boundary_margin)
        edit = [i < donor_boundary for i in range(NUM_LATENT_FRAMES)]
    return concept, edit


def edit_latent_reflected(
    latent: torch.Tensor, edit_mask: list[bool], region: str
) -> tuple[torch.Tensor, dict[int, list[int]]]:
    """Fill the frames marked True in ``edit_mask`` by mirroring the safe segment's motion inward
    from the boundary, bouncing back and forth across the safe frames once the far end is reached,
    instead of freezing a single donor frame across the whole block.

    ``frame_replace_ops.edit_latent``'s two-sided interpolation never actually engages for this
    construction — the concept block always touches a clip edge (one side of the split), so it
    always hits that function's one-sided fallback: every replaced frame becomes an exact copy of
    the single safe frame nearest the boundary. Per that function's own docstring, a hard
    single-frame copy taught the model to hold still and suppressed motion (exp055: concept -84%,
    unrelated -29%) — a real risk this construction hits on every clip, not an edge case. Playing
    the safe segment backward (frame N-1, N-2, ..., 0, 1, 2, ..., bouncing) instead gives the model
    a target with real, if mirrored, motion, sourced entirely from confirmed-safe frames.

    Position 0 in the fill order (nearest the boundary) maps to the safe frame immediately adjacent
    to it, so the seam itself stays a near-identical-content cut, same as the plain single-copy
    version — this only changes what happens *away* from the seam.
    """
    safe = [i for i, e in enumerate(edit_mask) if not e]
    fill = [i for i, e in enumerate(edit_mask) if e]
    if region == "second":
        safe_ordered = sorted(safe, reverse=True)  # nearest-to-boundary first
        fill_ordered = sorted(fill)  # nearest-to-boundary first
    else:
        safe_ordered = sorted(safe)
        fill_ordered = sorted(fill, reverse=True)

    n = len(safe_ordered)
    period = 2 * (n - 1) if n > 1 else 1
    edited = latent.clone()
    donor_map: dict[int, list[int]] = {}
    for k, pos in enumerate(fill_ordered):
        m = k % period
        idx = m if m < n else period - m
        donor = safe_ordered[idx]
        edited[:, :, pos] = latent[:, :, donor]
        donor_map[pos] = [donor]
    return edited, donor_map


def main(config: Config) -> None:
    latents_dir = os.path.join(config.output_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)
    videos_dir = os.path.join(config.output_dir, config.videos_subdir)
    if config.save_videos:
        os.makedirs(videos_dir, exist_ok=True)

    pipe = CogVideoXPipeline.from_pretrained(config.model_id, torch_dtype=DTYPE).to(DEVICE)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    assert pipe.scheduler.config.prediction_type == "v_prediction", (
        f"Expected v_prediction scheduler, got {pipe.scheduler.config.prediction_type!r}"
    )
    scaling_factor = float(pipe.vae.config.scaling_factor)

    detector = build_detector(config.concept, config.output_dir, config.concept_target)
    split_cfg = config.split_config()

    df = pd.read_csv(config.csv_path)
    for col in ("prompt_a", "prompt_b", "prompt_c", "seed"):
        if col not in df.columns:
            raise ValueError(f"{config.csv_path} missing required column '{col}'.")

    metadata: list[dict] = []
    skipped: list[dict] = []
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            seed = int(row["seed"])
            stem = f"p{idx}_s{seed}"
            sf, region = resolve_split(split_cfg, random.Random(seed))

            # The concept mask is known by construction (see module docstring): generate_split_clip
            # conditions frames [sf:] (region="second") or [:sf] (region="first") on prompt_a during
            # the split phase. No detection needed to find it.
            concept_latent, edit_mask = build_edit_masks(sf, region, config.boundary_margin)
            nofire = [i for i, is_c in enumerate(edit_mask) if not is_c]

            if len(nofire) < config.min_donor_frames:
                skipped.append({"stem": stem, "seed": seed, "reason": "insufficient_donor_frames",
                                "concept_region": region, "num_donor_latent_frames": len(nofire)})
            else:
                # 1. Manufacture the partial-concept clip (this is x0_original).
                z_bcfhw = generate_split_clip(
                    pipe, row["prompt_a"], row["prompt_b"], row["prompt_c"], seed, split_cfg, sf, region)
                assert z_bcfhw.shape == EXPECTED_LATENT_SHAPE, f"unexpected latent shape {z_bcfhw.shape}"

                # 2. Run the detector for logging only (frame_confidences below) — it does not
                # decide the mask or gate keep/skip, see module docstring.
                original_frames = decode_to_bgr_frames(pipe, z_bcfhw)
                confidences = detector.frame_confidences(original_frames)
                concept_pixel = [c >= config.frame_concept_threshold for c in confidences]

                # 3. Replace concept frames (+ boundary margin) with a reflected fill of the safe
                # segment's motion -> x0_edited (see edit_latent_reflected's docstring for why this
                # is used instead of frame_replace_ops.edit_latent's single-frame-copy fallback).
                x0_edited, donor_map = edit_latent_reflected(z_bcfhw, edit_mask, region)
                edited_confidences = detector.frame_confidences(decode_to_bgr_frames(pipe, x0_edited))

                edited_path = f"{stem}_x0edited.pt"
                original_path = f"{stem}_x0original.pt"
                torch.save(x0_edited.cpu(), os.path.join(latents_dir, edited_path))
                torch.save(z_bcfhw.cpu(), os.path.join(latents_dir, original_path))
                if config.save_videos:
                    write_mp4(original_frames, os.path.join(videos_dir, f"{stem}_original.mp4"))
                    write_mp4(decode_to_bgr_frames(pipe, x0_edited), os.path.join(videos_dir, f"{stem}_edited.mp4"))

                # variants["split"] always mirrors the flat top-level keys below unchanged, so every
                # dataset built before emit_whole_clip_target existed still loads exactly as before.
                variants: dict = {
                    "split": {
                        "latent_path": edited_path,
                        "original_latent_path": original_path,
                        "concept_latent_mask": concept_latent,
                    },
                }

                if config.emit_whole_clip_target:
                    # 4. (optional) A whole-clip target: prompt A's own plain clip is the "original",
                    # prompt B's same-seed plain clip is the donor/edited target. No donor-frame
                    # gate here — every frame is concept-bearing in A and concept-free in B by
                    # construction, unlike the frame-local split target above.
                    z_a = generate_plain_clip(pipe, row["prompt_a"], seed, split_cfg)
                    z_b = generate_plain_clip(pipe, row["prompt_b"], seed, split_cfg)
                    a_frames = decode_to_bgr_frames(pipe, z_a)
                    b_frames = decode_to_bgr_frames(pipe, z_b)
                    a_confidences = detector.frame_confidences(a_frames)
                    b_confidences = detector.frame_confidences(b_frames)

                    wholeclip_edited_path = f"{stem}_x0wholeclip.pt"
                    wholeclip_original_path = f"{stem}_x0wholeclip_original.pt"
                    torch.save(z_b.cpu(), os.path.join(latents_dir, wholeclip_edited_path))
                    torch.save(z_a.cpu(), os.path.join(latents_dir, wholeclip_original_path))
                    if config.save_videos:
                        write_mp4(a_frames, os.path.join(videos_dir, f"{stem}_wholeclip_a.mp4"))
                        write_mp4(b_frames, os.path.join(videos_dir, f"{stem}_wholeclip_b.mp4"))

                    variants["wholeclip"] = {
                        "latent_path": wholeclip_edited_path,
                        "original_latent_path": wholeclip_original_path,
                        "concept_latent_mask": [True] * NUM_LATENT_FRAMES,  # every frame is concept, by construction
                        "donor_prompt": row["prompt_b"],
                        "frame_confidences": [round(c, 4) for c in a_confidences],
                        "donor_frame_confidences": [round(c, 4) for c in b_confidences],
                    }

                metadata.append({
                    "prompt": row["prompt_a"],  # the plain concept prompt we erase at inference
                    "seed": seed,
                    "latent_path": edited_path,
                    "original_latent_path": original_path,
                    "concept": config.concept,
                    "concept_target": config.concept_target,
                    "concept_latent_mask": concept_latent,  # true construction mask (no margin)
                    "edited_latent_mask": edit_mask,  # what was actually replaced (concept + boundary_margin)
                    "concept_pixel_mask": concept_pixel,  # informational only, not used to build the mask
                    "concept_region": region,
                    "split_latent_frame": sf,
                    "split_step_frac": config.split_step_frac,
                    "tail_prompt_mode": config.tail_prompt_mode,
                    "split_mode": config.split_mode,
                    "concept_guidance_scale": config.concept_guidance_scale or config.guidance_scale,
                    "boundary_margin": config.boundary_margin,
                    "donor_map": {str(k): v for k, v in donor_map.items()},
                    "frame_confidences": [round(c, 4) for c in confidences],
                    "edited_frame_confidences": [round(c, 4) for c in edited_confidences],
                    "original_max_confidence": round(max(confidences), 4),
                    "edited_max_confidence": round(max(edited_confidences), 4),
                    "scaling_factor": scaling_factor,
                    "prediction_type": "v_prediction",
                    "variants": variants,
                })

            # Flush after every row (kept or skipped) so a crash/timeout never silently loses the
            # tail of the CSV — previously this write only ran on the kept path, so trailing skips
            # (e.g. the CSV's last rows) were recorded in memory but never reached skipped.json.
            with open(os.path.join(config.output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            with open(os.path.join(config.output_dir, "skipped.json"), "w") as f:
                json.dump(skipped, f, indent=2)

    print(f"frame_replace_split precompute done: {len(metadata)} targets, {len(skipped)} skipped -> {config.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="CSV with prompt_a/prompt_b/prompt_c/seed")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    main(Config(csv_path=args.csv_path, output_dir=args.output_dir))
