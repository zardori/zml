"""Per-video motion scoring (dense optical flow).

A guard against frame_replace's structural failure mode: the target is built by *copying* the
nearest concept-free frame over each concept frame, so a contiguous concept block becomes a frozen,
repeated segment. A model trained on that can learn to slow down or freeze the video on erased
prompts — a degradation that fire_detection_rate, CLIP, and colorfulness are all blind to (a frozen
clip can still be colorful, on-prompt, and fire-free).

The score is the mean Farneback optical-flow magnitude between consecutive frames, averaged over the
video. A frozen / repeated-frame video reads ~0; a normally-moving clip reads well above 0. Frames
are converted to grayscale and downscaled before the flow so the metric is cheap and resolution-
robust; the per-video score is the mean over frame pairs. Use it alongside colorfulness to tell
genuine erasure from a video that "erased" the concept by grinding to a halt.
"""

import argparse
import os

import cv2
import numpy as np
from zml.video_files import list_video_files

# Longest frame side (px) before optical flow; downscaling keeps the metric fast and makes the
# magnitude scale comparable across the fixed eval geometry.
FLOW_MAX_SIDE = 240
# Farneback dense-flow parameters (OpenCV defaults tuned for smooth, global motion).
FARNEBACK_KWARGS = dict(
    pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)


class VideoMotionScorer:
    """Mean dense optical-flow magnitude over a video's consecutive frame pairs.

    Per frame pair: grayscale Farneback flow (fx, fy), magnitude = √(fx² + fy²), averaged over
    pixels. The per-video score is the mean over all pairs. A single-frame or unreadable video
    scores 0.0.
    """

    def __init__(self, video_dir: str, max_side: int = FLOW_MAX_SIDE):
        self.video_dir = video_dir
        self.max_side = max_side

    def _prep(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = self.max_side / max(h, w)
        if scale < 1.0:
            gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        return gray

    def process_video(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        prev = None
        mags: list[float] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = self._prep(frame)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, gray, None, **FARNEBACK_KWARGS)
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                mags.append(float(mag.mean()))
            prev = gray
        cap.release()
        return float(np.mean(mags)) if mags else 0.0

    def process_videos(self) -> list[float]:
        """Returns the per-video mean motion magnitude for every video in video_dir."""
        video_files = list_video_files(self.video_dir)
        return [self.process_video(os.path.join(self.video_dir, f)) for f in video_files]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mean optical-flow motion for videos")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory with the videos")
    args = parser.parse_args()
    scores = VideoMotionScorer(video_dir=args.input_dir).process_videos()
    print({"motion_per_video": scores, "motion_score_mean": float(np.mean(scores)) if scores else 0.0})
