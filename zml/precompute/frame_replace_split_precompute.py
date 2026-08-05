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
from zml.precompute.split_prompt_precompute import generate_split_clip, resolve_split
from zml.precompute.split_prompt_precompute import Config as SplitConfig
from zml.unlearn.frame_replace_ops import (
    EXPECTED_LATENT_SHAPE,
    NUM_LATENT_FRAMES,
    NUM_PIXEL_FRAMES,
    decode_to_bgr_frames,
    edit_latent,
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
    # The concept latent mask is derived directly from (split_latent_frame, concept_region) — see
    # module docstring. The detector still runs to log per-frame confidences for human review, but
    # frame_concept_threshold no longer gates keep/skip.
    # Per-frame detector score above which a frame counts as containing the concept, for logging
    # only. The scale is detector-specific (NudeNet detection score, ResNet-50 class probability),
    # so it must be recalibrated whenever `concept` changes.
    frame_concept_threshold: float = 0.3
    frame_nudity_threshold: float | None = None  # deprecated alias, kept so exp061's config still loads
    min_donor_frames: int = 2  # skip clips whose known concept-free side has fewer latent frames
    concept: str = "nudity"  # selects the detector, see zml/benchmarks/registry.py
    concept_target: str | None = None  # target within a concept family, e.g. "chain saw" for "object"
    output_dir: str = "."
    save_videos: bool = True
    videos_subdir: str = "videos"

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
        )


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
            # the split phase. No detection needed to find it, and the check below is a cheap
            # pre-generation guard (not a GPU-time detector call) against a degenerate sf.
            concept_latent = [i >= sf for i in range(NUM_LATENT_FRAMES)] if region == "second" \
                else [i < sf for i in range(NUM_LATENT_FRAMES)]
            nofire = [i for i, is_c in enumerate(concept_latent) if not is_c]

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

                # 3. Replace concept frames with (interpolated) donors -> x0_edited.
                x0_edited, donor_map = edit_latent(z_bcfhw, concept_latent)
                edited_confidences = detector.frame_confidences(decode_to_bgr_frames(pipe, x0_edited))

                edited_path = f"{stem}_x0edited.pt"
                original_path = f"{stem}_x0original.pt"
                torch.save(x0_edited.cpu(), os.path.join(latents_dir, edited_path))
                torch.save(z_bcfhw.cpu(), os.path.join(latents_dir, original_path))
                if config.save_videos:
                    write_mp4(original_frames, os.path.join(videos_dir, f"{stem}_original.mp4"))
                    write_mp4(decode_to_bgr_frames(pipe, x0_edited), os.path.join(videos_dir, f"{stem}_edited.mp4"))

                metadata.append({
                    "prompt": row["prompt_a"],  # the plain concept prompt we erase at inference
                    "seed": seed,
                    "latent_path": edited_path,
                    "original_latent_path": original_path,
                    "concept": config.concept,
                    "concept_target": config.concept_target,
                    "concept_latent_mask": concept_latent,
                    "concept_pixel_mask": concept_pixel,  # informational only, not used to build the mask
                    "concept_region": region,
                    "split_latent_frame": sf,
                    "donor_map": {str(k): v for k, v in donor_map.items()},
                    "frame_confidences": [round(c, 4) for c in confidences],
                    "edited_frame_confidences": [round(c, 4) for c in edited_confidences],
                    "original_max_confidence": round(max(confidences), 4),
                    "edited_max_confidence": round(max(edited_confidences), 4),
                    "scaling_factor": scaling_factor,
                    "prediction_type": "v_prediction",
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
