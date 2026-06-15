"""Build edited-target latents for the ``frame_replace`` unlearning method.

For each (prompt, seed) the model generates a video, we detect which frames contain fire,
and we replace the fire-containing *latent* frames with the nearest fire-free ("donor") latent
frame from the same clip. The resulting edited clean latent ``x0_edited`` is the supervised
target the trainer fine-tunes toward (see ``zml/unlearn/unlearn_frame_replace.py``).

This is an offline step: generating + decoding + running the fire detector per training step
would be far too expensive, so we precompute the targets once and the trainer just loads them.

Outputs (``latents/``, ``metadata.json``, ``skipped.json``) go into ``output_dir`` — the same
per-run ``outputs_{timestamp}`` directory the training/eval entrypoints use. A training run that
wants this dataset just points at that directory's ``metadata.json`` / ``latents``.

Run standalone, e.g.:
    uv run python -m zml.precompute.frame_replace_precompute \
        --csv_path prompts/cogvideox_fire.csv --output_dir frame_replace_dataset
"""

import argparse
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from diffusers import CogVideoXPipeline
from tqdm import tqdm

from zml.eval.check_for_fire import VideoFireDetector

# Latent geometry for CogVideoX-5b at 49 frames / 480x720 (see unhype.py constants).
NUM_CHANNELS = 16
NUM_LATENT_FRAMES = 13
LATENT_HEIGHT = 60
LATENT_WIDTH = 90
EXPECTED_LATENT_SHAPE = (1, NUM_CHANNELS, NUM_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)

# CogVideoX 3D causal VAE temporal compression: latent frame 0 encodes a single pixel frame
# (the causal anchor), every later latent frame encodes TEMPORAL_RATIO pixel frames.
TEMPORAL_RATIO = 4
NUM_PIXEL_FRAMES = 1 + TEMPORAL_RATIO * (NUM_LATENT_FRAMES - 1)  # 49

DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    csv_path: str  # CSV with 'prompt' and 'seed' columns
    model_id: str = "THUDM/CogVideoX-5b"
    num_inference_steps: int = 50  # keep >=50 so the final latent is a clean x0
    guidance_scale: float = 6.0
    num_frames: int = NUM_PIXEL_FRAMES
    frame_fire_threshold: float = 0.5  # per-frame fire confidence above which a frame counts as fire
    min_nofire_frames: int = 2  # skip clips with fewer fire-free latent frames (avoids near-static targets)
    # Per-run outputs_{timestamp} dir (supplied by the thin entrypoint); receives latents/ + metadata.
    output_dir: str = "."


def latent_to_pixel_frames(latent_idx: int) -> list[int]:
    """Pixel-frame indices that fold into latent frame ``latent_idx`` under the 1+4k mapping."""
    if latent_idx == 0:
        return [0]
    start = 1 + TEMPORAL_RATIO * (latent_idx - 1)
    return list(range(start, start + TEMPORAL_RATIO))


def build_latent_fire_mask(fire_pixel: list[bool]) -> list[bool]:
    """A latent frame is "fire" if any of the pixel frames it encodes contains fire."""
    return [
        any(fire_pixel[p] for p in latent_to_pixel_frames(i))
        for i in range(NUM_LATENT_FRAMES)
    ]


def edit_latent(
    latent: torch.Tensor, fire_latent: list[bool]
) -> tuple[torch.Tensor, dict[int, int]]:
    """Replace each fire latent frame (along the F axis) with the nearest fire-free one."""
    nofire = [i for i, is_fire in enumerate(fire_latent) if not is_fire]
    edited = latent.clone()
    donor_map: dict[int, int] = {}
    for i in range(NUM_LATENT_FRAMES):
        if fire_latent[i]:
            donor = min(nofire, key=lambda j: abs(j - i))
            edited[:, :, i] = latent[:, :, donor]
            donor_map[i] = donor
    return edited, donor_map


def decode_to_bgr_frames(pipe: CogVideoXPipeline, latent_bcfhw: torch.Tensor) -> list[np.ndarray]:
    """Decode a clean latent to pixel frames as BGR uint8, matching the fire detector's input."""
    scaled = (1.0 / pipe.vae.config.scaling_factor) * latent_bcfhw
    decoded = pipe.vae.decode(scaled).sample  # (B, C, F, H, W) in ~[-1, 1]
    video = pipe.video_processor.postprocess_video(video=decoded, output_type="np")  # (B,F,H,W,C) [0,1]
    rgb_frames = (video[0] * 255.0).round().astype(np.uint8)
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb_frames]


def main(config: Config) -> None:
    latents_dir = os.path.join(config.output_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)

    pipe = CogVideoXPipeline.from_pretrained(config.model_id, torch_dtype=DTYPE).to(DEVICE)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    assert pipe.scheduler.config.prediction_type == "v_prediction", (
        f"Expected v_prediction scheduler, got {pipe.scheduler.config.prediction_type!r}"
    )
    scaling_factor = float(pipe.vae.config.scaling_factor)

    detector = VideoFireDetector(video_dir=config.output_dir)

    df = pd.read_csv(config.csv_path)
    metadata: list[dict] = []
    skipped: list[dict] = []
    donor_from_frame0 = 0

    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            prompt = row["prompt"]
            seed = int(row["seed"])

            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            out = pipe(
                prompt=prompt,
                num_frames=config.num_frames,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
                output_type="latent",
            )
            # output_type="latent" returns the scaled clean latent in (B, F, C, H, W) layout.
            z_bcfhw = out.frames.permute(0, 2, 1, 3, 4).contiguous()  # -> (B, C, F, H, W)
            assert z_bcfhw.shape == EXPECTED_LATENT_SHAPE, f"unexpected latent shape {z_bcfhw.shape}"

            frames = decode_to_bgr_frames(pipe, z_bcfhw)
            assert len(frames) == NUM_PIXEL_FRAMES, f"expected {NUM_PIXEL_FRAMES} frames, got {len(frames)}"

            confidences = detector.frame_fire_confidences(frames)
            fire_pixel = [c >= config.frame_fire_threshold for c in confidences]
            fire_latent = build_latent_fire_mask(fire_pixel)
            nofire = [i for i, is_fire in enumerate(fire_latent) if not is_fire]

            skip_reason = None
            if not any(fire_latent):
                skip_reason = "no_fire"
            elif len(nofire) < config.min_nofire_frames:
                skip_reason = "insufficient_donor_frames"
            if skip_reason is not None:
                skipped.append({"prompt": prompt, "seed": seed, "reason": skip_reason,
                                "num_nofire_latent_frames": len(nofire)})
                continue

            x0_edited, donor_map = edit_latent(z_bcfhw, fire_latent)
            if 0 in donor_map.values():
                donor_from_frame0 += 1

            latent_filename = f"p{idx}_s{seed}_x0edited.pt"
            torch.save(x0_edited.cpu(), os.path.join(latents_dir, latent_filename))
            metadata.append({
                "prompt": prompt,
                "seed": seed,
                "latent_path": latent_filename,
                "fire_pixel_mask": fire_pixel,
                "fire_latent_mask": fire_latent,
                "donor_map": {str(k): v for k, v in donor_map.items()},
                "frame_confidences": [round(c, 4) for c in confidences],
                "scaling_factor": scaling_factor,
                "prediction_type": "v_prediction",
            })

    with open(os.path.join(config.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(config.output_dir, "skipped.json"), "w") as f:
        json.dump(skipped, f, indent=2)

    print(f"Kept {len(metadata)} / {len(df)} clips ({len(skipped)} skipped). "
          f"Latent frame 0 used as a donor in {donor_from_frame0} clips.")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Build frame-replace edited-target latents.")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV with 'prompt' and 'seed' columns")
    parser.add_argument("--output_dir", type=str, default=Config.output_dir,
                        help="Output dir for latents/ + metadata.json")
    parser.add_argument("--model_id", type=str, default=Config.model_id)
    parser.add_argument("--num_inference_steps", type=int, default=Config.num_inference_steps,
                        help="Keep >=50 so the final latent is a clean x0")
    parser.add_argument("--guidance_scale", type=float, default=Config.guidance_scale)
    parser.add_argument("--num_frames", type=int, default=Config.num_frames)
    parser.add_argument("--frame_fire_threshold", type=float, default=Config.frame_fire_threshold,
                        help="Per-frame fire confidence above which a frame counts as fire")
    parser.add_argument("--min_nofire_frames", type=int, default=Config.min_nofire_frames,
                        help="Skip clips with fewer fire-free latent frames (avoids near-static targets)")
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    main(parse_args())
