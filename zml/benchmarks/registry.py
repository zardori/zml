"""One place that maps a config's ``concept`` string to its video detector.

Every detector exposes the same interface — ``frame_confidences(frames)``, ``score_video``,
``process_video``, ``process_videos()`` — and ``process_videos`` returns
``"<concept>_detection_rate"`` and ``"<concept>_area_score_mean"``, the keys ``zml/unlearn/eval.py``
derives from ``concept``. Callers therefore never branch on the concept themselves.

Imports are lazy on purpose: the fire path must not require ``nudenet``, the nudity path must not
require ``torchvision``, and neither should pull in a model the run will not use.
"""

from typing import Protocol

import numpy as np


class VideoDetector(Protocol):
    """The contract every concept detector satisfies."""

    def frame_confidences(self, frames: list[np.ndarray]) -> list[float]: ...
    def process_video(self, video_path: str) -> bool: ...
    def process_videos(self) -> dict[str, float]: ...


CONCEPTS = ("fire", "nudity", "object")


def build_detector(concept: str, video_dir: str, target: str | None = None, **kwargs) -> VideoDetector:
    """Detector for ``concept`` over ``video_dir``.

    ``target`` names the specific thing to detect for concepts that cover a family — currently only
    ``object``, where it is the ImageNet class (e.g. ``"chain saw"``).
    """
    if concept == "fire":
        from zml.eval.check_for_fire import VideoFireDetector

        return VideoFireDetector(video_dir=video_dir, **kwargs)
    if concept == "nudity":
        from zml.benchmarks.check_for_nudity import VideoNudeDetector

        return VideoNudeDetector(video_dir=video_dir, **kwargs)
    if concept == "object":
        from zml.benchmarks.check_for_object import VideoObjectDetector

        if not target:
            raise ValueError(
                "concept 'object' needs a target ImageNet class; set `concept_target` in the config."
            )
        return VideoObjectDetector(video_dir=video_dir, target_class=target, **kwargs)
    raise ValueError(f"Unknown concept {concept!r}; expected one of {CONCEPTS}.")
