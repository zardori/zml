"""Shared geometry constants and latent-editing ops for the ``frame_replace`` method.

These helpers are model-agnostic (they operate on already-generated latents / decoded frames)
and are reused by both the offline precompute step
(``zml/precompute/frame_replace_precompute.py``) and the online trainer
(``zml/unlearn/unlearn_frame_replace_online.py``).
"""

import cv2
import numpy as np
import torch
from diffusers import CogVideoXPipeline

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

VIDEO_FPS = 8  # CogVideoX default playback rate


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
) -> tuple[torch.Tensor, dict[int, list[int]]]:
    """Replace each fire latent frame (along the F axis) with a motion-preserving fire-free target.

    For a fire frame ``i`` bracketed by fire-free frames on both sides, linearly interpolate in latent
    space between the nearest fire-free frame *before* it (``lo``) and *after* it (``hi``):
    ``edited[i] = (1-w)*latent[lo] + w*latent[hi]`` with ``w = (i-lo)/(hi-lo)``. This ramps smoothly
    across a fire block instead of copying one frozen frame across the whole block — the hard copy
    taught the model to hold still and globally suppressed motion (exp055: concept -84%, unrelated
    -29%). Fire frames at the clip start/end have a fire-free neighbour on only one side and fall back
    to copying that nearest frame.

    Returns the edited latent and a ``donor_map`` mapping each fire frame to the fire-free endpoint
    frame index/indices used (two for an interpolated frame, one for a one-sided edge copy).
    """
    nofire = [i for i, is_fire in enumerate(fire_latent) if not is_fire]
    edited = latent.clone()
    donor_map: dict[int, list[int]] = {}
    for i in range(NUM_LATENT_FRAMES):
        if not fire_latent[i]:
            continue
        lo = max((j for j in nofire if j < i), default=None)
        hi = min((j for j in nofire if j > i), default=None)
        if lo is None and hi is None:
            continue  # no donors at all; callers guarantee >= min_nofire_frames, so unreachable
        if lo is None or hi is None:
            donor = hi if lo is None else lo  # edge block: only one side available -> copy it
            edited[:, :, i] = latent[:, :, donor]
            donor_map[i] = [donor]
        else:
            w = (i - lo) / (hi - lo)
            # Interpolate in float to avoid bf16 rounding, then cast back to the latent dtype.
            edited[:, :, i] = (
                (1.0 - w) * latent[:, :, lo].float() + w * latent[:, :, hi].float()
            ).to(latent.dtype)
            donor_map[i] = [lo, hi]
    return edited, donor_map


def decode_to_bgr_frames(pipe: CogVideoXPipeline, latent_bcfhw: torch.Tensor) -> list[np.ndarray]:
    """Decode a clean latent to pixel frames as BGR uint8, matching the fire detector's input."""
    scaled = (1.0 / pipe.vae.config.scaling_factor) * latent_bcfhw
    decoded = pipe.vae.decode(scaled).sample  # (B, C, F, H, W) in ~[-1, 1]
    video = pipe.video_processor.postprocess_video(video=decoded, output_type="np")  # (B,F,H,W,C) [0,1]
    rgb_frames = (video[0] * 255.0).round().astype(np.uint8)
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb_frames]


def write_mp4(frames_bgr: list[np.ndarray], path: str, fps: int = VIDEO_FPS) -> None:
    """Encode BGR uint8 frames to an MP4 at ``path``."""
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()
