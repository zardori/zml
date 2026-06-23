"""Build base-model preservation latents for the ``frame_replace`` retention anchor.

For each ``(prompt, seed)`` in a preservation CSV this generates a clip with the *frozen* base
model and saves its clean latent ``x0`` unchanged. The offline ``frame_replace`` trainer
(``zml/unlearn/unlearn_frame_replace.py``) then SFTs the LoRA toward these latents, anchoring the
preservation prompts to the base model's own output so erasing fire does not drift them.

Unlike ``frame_replace_precompute.py`` there is no fire detection, editing, or skipping — every
clip is kept as-is — so this is intentionally a separate, minimal script.

Run standalone, e.g.:
    uv run python -m zml.precompute.preservation_precompute \
        --csv_path prompts/cogvideox_fire_preservation.csv --output_dir preservation_dataset
"""

import argparse
import json
import os
from dataclasses import dataclass

import pandas as pd
import torch
from diffusers import CogVideoXPipeline
from tqdm import tqdm

from zml.unlearn.frame_replace_ops import (
    EXPECTED_LATENT_SHAPE,
    NUM_PIXEL_FRAMES,
    decode_to_bgr_frames,
    write_mp4,
)

DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    csv_path: str  # CSV with 'prompt' and 'seed' columns
    model_id: str = "THUDM/CogVideoX-5b"
    num_inference_steps: int = 50  # keep >=50 so the final latent is a clean x0
    guidance_scale: float = 6.0
    num_frames: int = NUM_PIXEL_FRAMES
    # Per-run outputs_{timestamp} dir (supplied by the thin entrypoint); receives latents/ + metadata.
    output_dir: str = "."
    save_videos: bool = False  # decode + write a sanity MP4 alongside each latent
    videos_subdir: str = "videos"


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

    df = pd.read_csv(config.csv_path)
    metadata: list[dict] = []

    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            prompt = row["prompt"]
            seed = int(row["seed"])
            stem = f"p{idx}_s{seed}"

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

            latent_filename = f"{stem}_x0.pt"
            torch.save(z_bcfhw.cpu(), os.path.join(latents_dir, latent_filename))

            entry = {
                "prompt": prompt,
                "seed": seed,
                "latent_path": latent_filename,
                "scaling_factor": scaling_factor,
                "prediction_type": "v_prediction",
            }
            if config.save_videos:
                video_path = os.path.join(videos_dir, f"{stem}.mp4")
                write_mp4(decode_to_bgr_frames(pipe, z_bcfhw), video_path)
                entry["video"] = os.path.relpath(video_path, config.output_dir)
            metadata.append(entry)

    with open(os.path.join(config.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(metadata)} preservation latents to {latents_dir}")
    if config.save_videos:
        print(f"Sanity videos written to {videos_dir}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Build base-model preservation latents.")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV with 'prompt' and 'seed' columns")
    parser.add_argument("--output_dir", type=str, default=Config.output_dir,
                        help="Output dir for latents/ + metadata.json")
    parser.add_argument("--model_id", type=str, default=Config.model_id)
    parser.add_argument("--num_inference_steps", type=int, default=Config.num_inference_steps,
                        help="Keep >=50 so the final latent is a clean x0")
    parser.add_argument("--guidance_scale", type=float, default=Config.guidance_scale)
    parser.add_argument("--num_frames", type=int, default=Config.num_frames)
    parser.add_argument("--videos_subdir", type=str, default=Config.videos_subdir)
    videos_group = parser.add_mutually_exclusive_group()
    videos_group.add_argument("--save_videos", dest="save_videos", action="store_true",
                              help="Decode + write a sanity MP4 alongside each latent")
    videos_group.add_argument("--skip_videos", dest="save_videos", action="store_false",
                              help="Skip video decode/encode entirely (default, faster)")
    parser.set_defaults(save_videos=Config.save_videos)
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    main(parse_args())
