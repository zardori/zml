"""Partial-fire objective: turn a per-frame fire-confidence profile into a quality score.

The ``frame_replace`` method needs clips where fire occupies *some* frames and other frames are
fire-free (donor frames). This module scores how cleanly a generated clip splits into a fire-free
part and a fire part — a single contiguous off->on transition near the middle is ideal, flicker is
not — and decides whether the clip is usable at all. It is deliberately pure (operates on a list of
per-frame confidences) so it can be unit-tested locally without a GPU.
"""

from dataclasses import dataclass

from zml.unlearn.frame_replace_ops import NUM_PIXEL_FRAMES, build_latent_fire_mask

# separation_score component weights (sum to 1.0). Tunable.
W_TRANSITION = 0.35  # reward a single off<->on boundary, punish flicker
W_CONTIGUITY = 0.25  # reward two large contiguous blocks (clean cut)
W_BALANCE = 0.25     # reward fire occupying ~half the clip (border in the middle)
W_MARGIN = 0.15      # reward confident fire vs. confident no-fire separation


@dataclass(frozen=True)
class ScoreThresholds:
    """Acceptance gates. Centeredness is *not* gated here — it only feeds separation_score."""

    frame_fire_threshold: float = 0.5  # per-frame conf to count as fire (matches frame_replace)
    min_nofire_latent_frames: int = 2  # donors needed by edit_latent (matches min_nofire_frames)
    clip_min: float = 0.22             # text-video alignment floor (quality gate)
    colorfulness_min: float = 8.0      # desaturation/degeneracy floor (quality gate)


@dataclass
class PartialFireMetrics:
    onset_frame: int | None
    offset_frame: int | None
    fire_fraction: float
    num_nofire_latent_frames: int
    num_transitions: int
    longest_nofire_run: int
    longest_fire_run: int
    balance: float
    confidence_margin: float
    separation_score: float
    clip_score: float
    colorfulness: float
    accepted: bool


def _longest_run(mask: list[bool], value: bool) -> int:
    best = run = 0
    for m in mask:
        run = run + 1 if m == value else 0
        best = max(best, run)
    return best


def _count_transitions(mask: list[bool]) -> int:
    return sum(1 for i in range(1, len(mask)) if mask[i] != mask[i - 1])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score(
    confidences: list[float],
    clip_score: float,
    colorfulness: float,
    thresholds: ScoreThresholds,
) -> PartialFireMetrics:
    fire_pixel = [c >= thresholds.frame_fire_threshold for c in confidences]
    fire_idx = [i for i, f in enumerate(fire_pixel) if f]
    nofire_latent = sum(1 for f in build_latent_fire_mask(fire_pixel) if not f)

    has_fire = bool(fire_idx)
    has_nofire = any(not f for f in fire_pixel)
    fire_fraction = len(fire_idx) / len(fire_pixel) if fire_pixel else 0.0

    longest_nofire = _longest_run(fire_pixel, False)
    longest_fire = _longest_run(fire_pixel, True)
    num_transitions = _count_transitions(fire_pixel)
    balance = max(0.0, 1.0 - abs(fire_fraction - 0.5) / 0.5)
    margin = _mean([confidences[i] for i in fire_idx]) - _mean(
        [confidences[i] for i in range(len(confidences)) if not fire_pixel[i]]
    )

    if has_fire and has_nofire:
        transition_term = 1.0 / num_transitions  # 1.0 for a single clean boundary
        contiguity_term = (longest_nofire + longest_fire) / NUM_PIXEL_FRAMES
        margin_term = min(max(margin, 0.0), 1.0)
        separation = (
            W_TRANSITION * transition_term
            + W_CONTIGUITY * contiguity_term
            + W_BALANCE * balance
            + W_MARGIN * margin_term
        )
    else:
        separation = 0.0  # all-fire or all-no-fire clips are unusable for frame_replace

    accepted = (
        has_fire
        and nofire_latent >= thresholds.min_nofire_latent_frames
        and clip_score >= thresholds.clip_min
        and colorfulness >= thresholds.colorfulness_min
    )

    return PartialFireMetrics(
        onset_frame=fire_idx[0] if fire_idx else None,
        offset_frame=fire_idx[-1] if fire_idx else None,
        fire_fraction=fire_fraction,
        num_nofire_latent_frames=nofire_latent,
        num_transitions=num_transitions,
        longest_nofire_run=longest_nofire,
        longest_fire_run=longest_fire,
        balance=balance,
        confidence_margin=margin,
        separation_score=separation,
        clip_score=clip_score,
        colorfulness=colorfulness,
        accepted=accepted,
    )
