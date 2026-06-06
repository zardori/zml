"""Per-video colorfulness scoring (Hasler–Süsstrunk).

A cheap, reliable guard against the desaturation failure mode: a model that "erases" a concept by
collapsing toward grayscale scores ~0 here, even when the fire detector reads it as "no fire" and
CLIP stays high. Use it alongside fire_detection_rate to tell genuine erasure from quality collapse.
"""

import argparse
import os

import cv2
import numpy as np


class VideoColorfulnessScorer:
    """Hasler–Süsstrunk colorfulness averaged over a video's frames.

    Per frame: rg = R − G, yb = ½(R + G) − B, and
    colorfulness = √(σ_rg² + σ_yb²) + 0.3·√(μ_rg² + μ_yb²). A grayscale frame has R = G = B, so
    rg = yb = 0 and colorfulness = 0. The per-video score is the mean over frames.
    """

    MEAN_WEIGHT = 0.3

    def __init__(self, video_dir: str):
        self.video_dir = video_dir

    @staticmethod
    def _frame_colorfulness(frame_bgr: np.ndarray) -> float:
        b, g, r = (c.astype(np.float32) for c in cv2.split(frame_bgr))
        rg = r - g
        yb = 0.5 * (r + g) - b
        std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
        return float(std_root + VideoColorfulnessScorer.MEAN_WEIGHT * mean_root)

    def process_video(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        scores: list[float] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            scores.append(self._frame_colorfulness(frame))
        cap.release()
        return float(np.mean(scores)) if scores else 0.0

    def process_videos(self) -> list[float]:
        """Returns the per-video mean colorfulness for every video in video_dir."""
        video_files = sorted(
            f for f in os.listdir(self.video_dir) if f.endswith((".mp4", ".avi", ".mov"))
        )
        return [self.process_video(os.path.join(self.video_dir, f)) for f in video_files]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Hasler–Süsstrunk colorfulness for videos")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory with the videos")
    args = parser.parse_args()
    scores = VideoColorfulnessScorer(video_dir=args.input_dir).process_videos()
    print({"colorfulness_per_video": scores, "colorfulness_mean": float(np.mean(scores)) if scores else 0.0})
