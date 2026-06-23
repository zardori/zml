from ultralytics import YOLO
import cv2
import numpy as np
import os
import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_THRESHOLD = 0.75
CANDIDATE_THRESHOLD = 0.5
TOP_K = 10


@dataclass
class VideoFireStats:
    """Per-video fire metrics derived from a single detection pass.

    ``detected`` is the binary top-k decision (unchanged semantics). The ``area_score``
    fields are a *continuous* fire-magnitude signal: each frame contributes the largest
    ``box_area_fraction * confidence`` among its fire boxes, so the score falls as the fire
    shrinks in area or confidence even while ``detected`` stays True. Area is normalised by
    the frame size, so the score is resolution-independent.
    """

    detected: bool
    area_score_sum: float  # sum of per-frame magnitudes over all frames
    area_score_mean: float  # area_score_sum / num_frames
    num_frames: int

class VideoFireDetector:
    # Pretrained fire-detection YOLOv8 weights (fire / smoke classes)
    MODEL_ID = "SalahALHaismawi/yolov26-fire-detection"
    # class 0 = fire, class 1 = smoke, class 2 = other
    CLASS_NAMES = {0: "fire", 1: "smoke", 2: "other"}
    FIRE_CLASS = 0

    def __init__(
        self,
        video_dir: str,
        conf_threshold: float = DEFAULT_THRESHOLD,
        candidate_threshold: float = CANDIDATE_THRESHOLD,
        top_k: int = TOP_K,
        debug: bool = False,
        debug_dir: str | None = None,
    ):
        model_path = hf_hub_download(
            repo_id=self.MODEL_ID,
            filename="best.pt"
        )
        self.model = YOLO(model_path)
        self.video_dir = video_dir
        self.conf_threshold = conf_threshold
        self.candidate_threshold = candidate_threshold
        self.top_k = top_k
        self.debug = debug
        if debug_dir is None:
            video_dir_path = Path(video_dir).resolve()
            debug_dir = str(video_dir_path.parent / f"{video_dir_path.name}_debug")
        self.debug_dir = debug_dir
        print("VideoFireDetector has been setup")

    def score_video(self, video_path: str) -> VideoFireStats:
        """Run the detector once over a video and derive both fire metrics.

        Per frame we take the max fire-box confidence (for the binary top-k decision, whose
        semantics are unchanged) and the max ``box_area_fraction * confidence`` (for the
        continuous area score). Both use ``candidate_threshold`` as the per-box gate and a
        per-frame max to avoid double-counting overlapping boxes.

        In debug mode, writes an annotated video with bounding boxes and per-frame
        confidence scores to self.debug_dir.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        writer: cv2.VideoWriter | None = None
        if self.debug:
            Path(self.debug_dir).mkdir(parents=True, exist_ok=True)
            debug_output_path = str(Path(self.debug_dir) / Path(video_path).name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(debug_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            print(f"Debug output: {debug_output_path}")

        conf_scores: list[float] = []  # nonzero per-frame max confidences (for the binary decision)
        area_score_sum = 0.0  # running sum of per-frame max(area_fraction * conf) over all frames
        num_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=self.candidate_threshold, classes=[self.FIRE_CLASS], verbose=False)
            boxes = [box for result in results for box in result.boxes]
            # Per-frame max avoids double-counting overlapping boxes on the same frame.
            frame_max = max((float(box.conf[0]) for box in boxes), default=0.0)
            # xywhn is the box [x, y, w, h] normalised to [0, 1]; w*h is its area fraction, so
            # area*conf shrinks when the fire covers less of the frame or is detected less
            # confidently — the signal we want once the binary rate saturates.
            frame_area = max(
                (float(box.xywhn[0][2]) * float(box.xywhn[0][3]) * float(box.conf[0]) for box in boxes),
                default=0.0,
            )
            if frame_max > 0:
                conf_scores.append(frame_max)
            area_score_sum += frame_area

            if self.debug:
                if frame_max > 0:
                    print(f"  frame {num_frames}: fire conf={frame_max:.3f} area*conf={frame_area:.4f}")
                annotated = results[0].plot()
                writer.write(annotated)

            num_frames += 1

        cap.release()
        if writer is not None:
            writer.release()

        detected = (
            len(conf_scores) >= self.top_k
            and sum(sorted(conf_scores, reverse=True)[:self.top_k]) / self.top_k >= self.conf_threshold
        )
        return VideoFireStats(
            detected=detected,
            area_score_sum=area_score_sum,
            area_score_mean=area_score_sum / num_frames if num_frames else 0.0,
            num_frames=num_frames,
        )

    def process_video(self, video_path: str) -> bool:
        """Binary fire decision for a single video (see ``score_video`` for the metric pass)."""
        return self.score_video(video_path).detected

    def frame_fire_confidences(self, frames: list[np.ndarray]) -> list[float]:
        """Return the max fire-class confidence per frame, aligned to every frame index.

        Unlike ``process_video`` (which drops zero-score frames and collapses to a single
        boolean), this keeps one score per input frame so callers can build a per-frame fire
        mask. Frames must be BGR uint8 — the same format ``cv2.VideoCapture`` feeds the model
        in ``process_video`` — so detection confidences stay comparable across both paths.
        """
        scores: list[float] = []
        for frame in frames:
            results = self.model(frame, conf=self.candidate_threshold, classes=[self.FIRE_CLASS], verbose=False)
            frame_max = max(
                (float(box.conf[0]) for result in results for box in result.boxes),
                default=0.0,
            )
            scores.append(frame_max)
        return scores

    def process_videos(self) -> dict[str, float]:
        """Returns the concept detection rate (CDR) and mean fire-area score over all videos."""
        video_files = [
            f for f in os.listdir(self.video_dir)
            if f.endswith((".mp4", ".avi", ".mov"))
        ]

        if not video_files:
            print(f"No video files found in {self.video_dir}")
            return {"fire_detection_rate": 0.0, "fire_area_score_mean": 0.0}

        fire_count = 0
        area_score_means: list[float] = []
        for video_name in video_files:
            stats = self.score_video(os.path.join(self.video_dir, video_name))
            area_score_means.append(stats.area_score_mean)
            if stats.detected:
                print("fire detected in", video_name)
                fire_count += 1

        return {
            "fire_detection_rate": fire_count / len(video_files),
            "fire_area_score_mean": float(np.mean(area_score_means)),
            "videos_with_fire": fire_count,
            "total_videos": len(video_files),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for fire in videos using a YOLOv8 fire-detection model")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory where the videos are saved")
    parser.add_argument("--conf_threshold", type=float, default=DEFAULT_THRESHOLD, help="Averaging threshold for final fire decision")
    parser.add_argument("--candidate_threshold", type=float, default=CANDIDATE_THRESHOLD, help="Minimum per-frame confidence to count as a candidate detection")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Number of top candidate frames required and averaged for final decision")
    parser.add_argument("--debug", action="store_true", help="Write annotated debug videos with fire bounding boxes and confidence scores")
    parser.add_argument("--debug_dir", type=str, default=None, help="Directory for debug videos (default: <input_dir>_debug sibling)")
    args = parser.parse_args()

    detector = VideoFireDetector(
        video_dir=args.input_dir,
        conf_threshold=args.conf_threshold,
        candidate_threshold=args.candidate_threshold,
        top_k=args.top_k,
        debug=args.debug,
        debug_dir=args.debug_dir,
    )
    scores = detector.process_videos()
    print(scores)
