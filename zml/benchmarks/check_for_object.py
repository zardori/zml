"""Per-frame + per-video detection of one ImageNet object class, mirroring ``check_for_nudity.py``.

Same four-method interface as the fire and nudity detectors, so this drops into the live-eval path
(``zml/unlearn/eval.py``) and the frame_replace dataset builder without either of them special-casing
objects:

- ``frame_confidences(frames)`` -> one score per frame, for building a per-frame concept mask.
- ``score_video`` / ``process_video`` / ``process_videos`` -> per-video stats + a detection rate.

Unlike YOLO-fire and NudeNet this is a *classifier*, not a box detector, so there is no area to
measure. The per-frame score is the softmax probability of the target class, and the
``object_area_score_mean`` key that ``evaluate()`` expects from every detector carries that mean
probability instead — the interface's generic "confidence mass" slot. ``object_prob_mean`` is the
same number under an honest name; prefer it when reading results.

``process_videos`` also reports the two numbers the published protocol is defined on:
``object_top1_accuracy`` and ``object_top5_accuracy``, pooled over every frame of every clip in the
directory. With ``restrict_to`` set it reports the same two under the restricted ranking convention
as well, from the same forward pass (see ``docs/imagenet_objects.md`` §3.1).
"""

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from zml.benchmarks.imagenet_classes import IMAGENETTE_INDICES, class_index
from zml.benchmarks.imagenet_classifier import DEFAULT_TOP_K, ImageNetFrameClassifier

# Fraction of a clip's frames that must be classified as the target for the clip to count as
# "contains the object". 0.5 keeps the binary rate readable next to fire/nudity detection rates.
DETECTION_THRESHOLD = 0.5


def read_bgr_frames(video_path: str) -> list[np.ndarray]:
    """Decode a video file to a list of BGR uint8 frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


@dataclass
class VideoObjectStats:
    """Per-video object metrics from a single classification pass (mirrors ``VideoNudeStats``)."""

    detected: bool  # top1_rate >= detection_threshold
    top1_rate: float  # fraction of frames whose top-1 prediction is the target class
    top5_rate: float  # fraction of frames with the target class in the top-5
    prob_mean: float  # mean per-frame softmax probability of the target class
    num_frames: int
    # Same two rates ranked within `restrict_to` only; None when the detector has no subset.
    top1_rate_restricted: float | None = None
    top5_rate_restricted: float | None = None


@dataclass
class _PooledSums:
    """Frame-count-weighted running totals, so accuracies pool over frames rather than over clips.

    Identical to a per-clip average for equal-length clips, but correct if one is ever truncated.
    """

    frames: int = 0
    top1: float = 0.0
    top5: float = 0.0
    prob: float = 0.0
    top1_restricted: float = 0.0
    top5_restricted: float = 0.0

    def add(self, stats: VideoObjectStats) -> None:
        self.frames += stats.num_frames
        self.top1 += stats.top1_rate * stats.num_frames
        self.top5 += stats.top5_rate * stats.num_frames
        self.prob += stats.prob_mean * stats.num_frames
        if stats.top1_rate_restricted is not None:
            self.top1_restricted += stats.top1_rate_restricted * stats.num_frames
            self.top5_restricted += stats.top5_rate_restricted * stats.num_frames

    def mean(self, total: float) -> float:
        return total / self.frames if self.frames else 0.0


class VideoObjectDetector:
    def __init__(
        self,
        video_dir: str,
        target_class: str,
        detection_threshold: float = DETECTION_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        classifier: ImageNetFrameClassifier | None = None,
        restrict_to: Sequence[int] | None = None,
    ):
        self.video_dir = video_dir
        self.target_class = target_class
        self.target_index = class_index(target_class)
        self.detection_threshold = detection_threshold
        self.top_k = top_k
        self.restrict_to = restrict_to
        # Sharing one classifier across directories avoids reloading ResNet-50 per target class.
        self.classifier = classifier or ImageNetFrameClassifier()
        print(f"VideoObjectDetector has been setup for {target_class!r} (index {self.target_index})")

    def frame_confidences(self, frames: list[np.ndarray]) -> list[float]:
        """Target-class probability per frame, aligned to every frame index (for a per-frame mask).

        Frames must be BGR uint8, like ``frame_fire_confidences`` / ``frame_nudity_confidences``.
        """
        if not frames:
            return []
        return self.classifier.classify(frames, self.target_index, self.top_k).target_probs.tolist()

    def score_video(self, video_path: str) -> VideoObjectStats:
        frames = read_bgr_frames(video_path)
        if not frames:
            return VideoObjectStats(detected=False, top1_rate=0.0, top5_rate=0.0, prob_mean=0.0, num_frames=0)

        # One forward pass, ranked under each convention we were asked for.
        probs = self.classifier.probs(frames)
        top1_rate, top5_rate = self._hit_rates(probs, restrict_to=None)
        restricted = (
            self._hit_rates(probs, restrict_to=self.restrict_to)
            if self.restrict_to is not None
            else (None, None)
        )
        return VideoObjectStats(
            detected=top1_rate >= self.detection_threshold,
            top1_rate=top1_rate,
            top5_rate=top5_rate,
            prob_mean=float(probs[:, self.target_index].mean()),
            num_frames=len(frames),
            top1_rate_restricted=restricted[0],
            top5_rate_restricted=restricted[1],
        )

    def _hit_rates(self, probs: torch.Tensor, restrict_to: Sequence[int] | None) -> tuple[float, float]:
        """Fraction of frames with the target class ranked first, and within the top-k."""
        topk = self.classifier.topk_indices(probs, self.top_k, restrict_to)
        return (
            float((topk[:, 0] == self.target_index).mean()),
            float((topk == self.target_index).any(axis=1).mean()),
        )

    def process_video(self, video_path: str) -> bool:
        """Binary object decision for a single video."""
        return self.score_video(video_path).detected

    def process_videos(self) -> dict[str, float]:
        """Detection rate + frame-pooled top-1/top-5 accuracy over all videos in ``video_dir``."""
        video_files = [f for f in os.listdir(self.video_dir) if f.endswith((".mp4", ".avi", ".mov"))]
        if not video_files:
            print(f"No video files found in {self.video_dir}")
            return self._summary(_PooledSums(), detected_count=0, total_videos=0)

        sums = _PooledSums()
        detected_count = 0
        for video_name in sorted(video_files):
            stats = self.score_video(os.path.join(self.video_dir, video_name))
            sums.add(stats)
            if stats.detected:
                print(f"{self.target_class} detected in {video_name}")
                detected_count += 1

        return self._summary(sums, detected_count, len(video_files))

    def _summary(self, sums: _PooledSums, detected_count: int, total_videos: int) -> dict[str, float]:
        prob_mean = sums.mean(sums.prob)
        summary = {
            "object_detection_rate": detected_count / total_videos if total_videos else 0.0,
            # Generic "confidence mass" slot required by evaluate(); same value as object_prob_mean.
            "object_area_score_mean": prob_mean,
            "object_prob_mean": prob_mean,
            "object_top1_accuracy": sums.mean(sums.top1),
            "object_top5_accuracy": sums.mean(sums.top5),
            "videos_with_object": detected_count,
            "total_videos": total_videos,
        }
        if self.restrict_to is not None:
            summary["object_top1_accuracy_restricted"] = sums.mean(sums.top1_restricted)
            summary["object_top5_accuracy_restricted"] = sums.mean(sums.top5_restricted)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify videos for one ImageNet object class")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory where the videos are saved")
    parser.add_argument("--target_class", type=str, required=True, help="e.g. 'chain saw' (see imagenet_classes.py)")
    parser.add_argument("--detection_threshold", type=float, default=DETECTION_THRESHOLD,
                        help="Fraction of frames that must be top-1 the target for a clip to count")
    parser.add_argument("--restricted", action="store_true",
                        help="Also report top-1/top-5 ranked within the ten protocol classes only")
    args = parser.parse_args()

    detector = VideoObjectDetector(
        video_dir=args.input_dir,
        target_class=args.target_class,
        detection_threshold=args.detection_threshold,
        restrict_to=IMAGENETTE_INDICES if args.restricted else None,
    )
    print(detector.process_videos())
