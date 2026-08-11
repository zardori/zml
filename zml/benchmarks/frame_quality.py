"""Detect generation-time failures (blank/structureless frames), independent of concept detection.

A concept detector answering "no face/fire/object here" is a real measurement when the frame
rendered something and the concept genuinely isn't in it. It is not a real measurement when the
frame never rendered at all — e.g. CogVideoX occasionally emits a solid-black or otherwise
structureless clip (bf16/VAE-tiling numerical failure, not a caught exception: generation succeeds,
``export_to_video`` writes a normal-looking non-empty file, nothing in the pipeline notices). Averaging
that into a metric silently mixes "the model didn't render this identity/concept" with "the model
didn't render anything", which is a different claim.

Found via exp090 (``docs/face_identity.md``): 11 of 150 base-model clips had degenerate frames (7
fully black), affecting `face_present_rate` and the `quality` block unevenly across identities.

**Deliberately a structure test, not a brightness test.** Many legitimate frames are very dark
without being blank — one exp090 clip (``donald_trump/video_25``, prompt: "...dimly lit...") has
mean luma 17.6, darker than several genuinely-degenerate clips, but the subject is clearly visible
and its minimum per-frame pixel std (10.73) sits well clear of ``DEGENERATE_FRAME_STD``. Standard
deviation of pixel intensity is ~0 for a constant frame regardless of how bright or dark that
constant is, so it does not confuse "dark" with "blank".

**Known limitation**, not fixed by this module: one exp090 clip
(``queen_elizabeth_ii/video_17``) is corrupted differently — two flat colour bands, not a single
constant value — so it has high pixel std despite carrying no real content. No cheap per-frame
statistic separates that case from a legitimately busy frame without risking false positives on real
clips; it is a human-review / DOVER catch, the same policy ``docs/imagenet_objects.md`` §3.1 follows
for the ranking-convention ambiguity rather than chasing every edge case with the detector.
"""

import numpy as np

# Calibrated 2026-08-11 against exp090's 150 base-model clips (see docs/face_identity.md): every
# frame of the 11 known-degenerate clips scores below this value (fully-black clips score exactly
# 0.0; the noisiest partial clip peaks at 2.45), while the darkest known-legitimate clip's minimum
# is 10.73 -- a >4x margin on both sides. A full scan of all 150 clips found zero frames outside the
# known-bad set below this threshold, so it is not overfit to the handful of cases used to pick it.
DEGENERATE_FRAME_STD = 5.0


def is_degenerate_frame(frame: np.ndarray) -> bool:
    """True if ``frame`` (HxWxC, any numeric dtype) has (near-)zero spatial structure.

    A blank/constant frame regardless of its brightness -- see the module docstring for why this is
    a structure test rather than a brightness test.
    """
    return float(frame.std()) < DEGENERATE_FRAME_STD


def degenerate_frame_mask(frames: list[np.ndarray]) -> list[bool]:
    """Per-frame degenerate flags, same order and length as ``frames``."""
    return [is_degenerate_frame(f) for f in frames]
