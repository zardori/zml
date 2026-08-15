"""Which numbers a given concept's runs are actually judged on.

Every eval writes ~30 keys per prompt set, and a deck that shows 30 columns shows nothing. This is the
one place that decides which handful is the headline for each concept, what to call it, and which
direction is better — so the collector, the tables and the sparklines all agree.

Direction matters more than it looks: erasure metrics go down and utility metrics go up, and a deck
that renders both as bare numbers invites the reader to congratulate a run for collapsing.
"""
from __future__ import annotations

from dataclasses import dataclass

from zml.results_io import DOVER_KEYS


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    decimals: int
    lower_is_better: bool | None  # None for metrics that are descriptive rather than scored

    def format(self, value: float | None) -> str | None:
        return None if value is None else f"{value:.{self.decimals}f}"


# Utility/quality metrics every concept records, shown after the concept's own erasure metrics.
SHARED_METRICS = (
    Metric("clip_score_mean", "CLIP", 2, False),
    Metric("colorfulness_mean", "Colourfulness", 1, None),
    Metric("motion_score_mean", "Motion", 3, False),
    Metric("dover_technical_mean", "DOVER-t", 4, False),
    Metric("dover_aesthetic_mean", "DOVER-a", 4, False),
)

# The erasure metric each concept is judged on, most load-bearing first.
#
# nudity leads with the *frame* rate, not the video rate: T2VUnlearning (2505.17550) reports per-frame
# and the video-level rate was shown unable to rank checkpoints — see docs/comparability_t2vunlearning.md.
CONCEPT_METRICS: dict[str, tuple[Metric, ...]] = {
    "nudity": (
        Metric("nudity_frame_rate", "Nudity (frame)", 4, True),
        Metric("nudity_detection_rate", "Nudity (video)", 3, True),
        Metric("nudity_area_score_mean", "Nudity area", 4, True),
    ),
    "fire": (
        Metric("fire_detection_rate", "Fire rate", 3, True),
        Metric("fire_area_score_mean", "Fire area", 4, True),
    ),
    "imagenet": (
        Metric("object_top1_accuracy", "Top-1", 4, True),
        Metric("object_top5_accuracy", "Top-5", 4, True),
        Metric("object_detection_rate", "Object rate", 3, True),
    ),
    "face": (
        Metric("face_id_similarity_mean", "ID similarity", 4, True),
        Metric("face_present_rate", "Face present", 3, None),
        Metric("degenerate_frame_rate", "Degenerate", 3, True),
    ),
}

# The generic aliases zml/unlearn/eval.py writes alongside the concept-specific names, used when a run
# predates the concept's own keys.
FALLBACK_METRICS = (
    Metric("concept_detection_rate", "Concept rate", 3, True),
    Metric("concept_area_score_mean", "Concept area", 4, True),
)

# Standalone eval headlines. ESR/PSR: erasure and preservation success rates, both percentages.
ESR_PSR_METRICS = (
    Metric("ESR-1", "ESR-1", 2, False),
    Metric("ESR-5", "ESR-5", 2, False),
    Metric("PSR-1", "PSR-1", 2, False),
    Metric("PSR-5", "PSR-5", 2, False),
)
ID_SIMILARITY_METRICS = (
    Metric("Erase", "ID sim (erased)", 4, True),
    Metric("Preserve", "ID sim (preserved)", 4, False),
)

# Prompt sets in the order a reader compares them: what was erased, then what must survive.
PROMPT_SET_ORDER = ("concept", "related", "unrelated", "anchor")


def headline_metrics(concept: str, present_keys: set[str]) -> list[Metric]:
    """The metrics to show for a concept, restricted to the ones this run actually recorded.

    DOVER is dropped when it reads 0.0 — that is helios' "not measured", never a score
    (CLAUDE.md, Metrics Logging).
    """
    concept_metrics = CONCEPT_METRICS.get(concept, ())
    if not any(metric.key in present_keys for metric in concept_metrics):
        concept_metrics = FALLBACK_METRICS
    return [m for m in (*concept_metrics, *SHARED_METRICS) if m.key in present_keys]


def measured_keys(group_scores: dict[str, float]) -> set[str]:
    """Scalar metric names in a prompt set's scores, excluding unmeasured DOVER zeros."""
    return {
        key
        for key, value in group_scores.items()
        if isinstance(value, (int, float)) and not (key in DOVER_KEYS and not value)
    }


def ordered_prompt_sets(scores: dict[str, dict]) -> list[str]:
    known = [name for name in PROMPT_SET_ORDER if name in scores]
    return known + sorted(name for name in scores if name not in PROMPT_SET_ORDER)
