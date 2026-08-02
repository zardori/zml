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
directory (see ``docs/imagenet_objects.md``).
"""

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np

from zml.benchmarks.imagenet_classes import class_index
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


class VideoObjectDetector:
    def __init__(
        self,
        video_dir: str,
        target_class: str,
        detection_threshold: float = DETECTION_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        classifier: ImageNetFrameClassifier | None = None,
    ):
        self.video_dir = video_dir
        self.target_class = target_class
        self.target_index = class_index(target_class)
        self.detection_threshold = detection_threshold
        self.top_k = top_k
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

        result = self.classifier.classify(frames, self.target_index, self.top_k)
        top1_rate = float((result.topk_indices[:, 0] == self.target_index).mean())
        top5_rate = float((result.topk_indices == self.target_index).any(axis=1).mean())
        return VideoObjectStats(
            detected=top1_rate >= self.detection_threshold,
            top1_rate=top1_rate,
            top5_rate=top5_rate,
            prob_mean=float(result.target_probs.mean()),
            num_frames=len(frames),
        )

    def process_video(self, video_path: str) -> bool:
        """Binary object decision for a single video."""
        return self.score_video(video_path).detected

    def process_videos(self) -> dict[str, float]:
        """Detection rate + frame-pooled top-1/top-5 accuracy over all videos in ``video_dir``."""
        video_files = [f for f in os.listdir(self.video_dir) if f.endswith((".mp4", ".avi", ".mov"))]
        if not video_files:
            print(f"No video files found in {self.video_dir}")
            return {"object_detection_rate": 0.0, "object_area_score_mean": 0.0, "object_prob_mean": 0.0,
                    "object_top1_accuracy": 0.0, "object_top5_accuracy": 0.0,
                    "videos_with_object": 0, "total_videos": 0}

        detected_count = 0
        total_frames = 0
        top1_hits = 0.0
        top5_hits = 0.0
        prob_sum = 0.0
        for video_name in sorted(video_files):
            stats = self.score_video(os.path.join(self.video_dir, video_name))
            # Weight by frame count so the accuracies are pooled over frames, not averaged over clips
            # (identical for equal-length clips, but correct if a clip is ever truncated).
            total_frames += stats.num_frames
            top1_hits += stats.top1_rate * stats.num_frames
            top5_hits += stats.top5_rate * stats.num_frames
            prob_sum += stats.prob_mean * stats.num_frames
            if stats.detected:
                print(f"{self.target_class} detected in {video_name}")
                detected_count += 1

        prob_mean = prob_sum / total_frames if total_frames else 0.0
        return {
            "object_detection_rate": detected_count / len(video_files),
            # Generic "confidence mass" slot required by evaluate(); same value as object_prob_mean.
            "object_area_score_mean": prob_mean,
            "object_prob_mean": prob_mean,
            "object_top1_accuracy": top1_hits / total_frames if total_frames else 0.0,
            "object_top5_accuracy": top5_hits / total_frames if total_frames else 0.0,
            "videos_with_object": detected_count,
            "total_videos": len(video_files),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify videos for one ImageNet object class")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory where the videos are saved")
    parser.add_argument("--target_class", type=str, required=True, help="e.g. 'chain saw' (see imagenet_classes.py)")
    parser.add_argument("--detection_threshold", type=float, default=DETECTION_THRESHOLD,
                        help="Fraction of frames that must be top-1 the target for a clip to count")
    args = parser.parse_args()

    detector = VideoObjectDetector(
        video_dir=args.input_dir,
        target_class=args.target_class,
        detection_threshold=args.detection_threshold,
    )
    print(detector.process_videos())
