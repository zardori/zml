from ultralytics import YOLO
import cv2
import os
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_THRESHOLD = 0.75
CANDIDATE_THRESHOLD = 0.5
TOP_K = 10

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

    def process_video(self, video_path: str) -> bool:
        """Returns True if fire is consistently detected across frames.

        Collects the per-frame max confidence score for frames where any fire box
        exceeds candidate_threshold, then averages the top-K scores. Returns True
        only if at least top_k candidate frames exist and their average meets
        conf_threshold. This reduces false positives from outlier frames.

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

        frame_scores: list[float] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=self.candidate_threshold, classes=[self.FIRE_CLASS], verbose=False)
            # Per-frame max avoids double-counting overlapping boxes on the same frame
            frame_max = max(
                (float(box.conf[0]) for result in results for box in result.boxes),
                default=0.0,
            )
            if frame_max > 0:
                frame_scores.append(frame_max)

            if self.debug:
                if frame_max > 0:
                    print(f"  frame {frame_idx}: fire conf={frame_max:.3f}")
                annotated = results[0].plot()
                writer.write(annotated)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        if len(frame_scores) < self.top_k:
            return False
        top_k_avg = sum(sorted(frame_scores, reverse=True)[:self.top_k]) / self.top_k
        return top_k_avg >= self.conf_threshold

    def process_videos(self) -> dict[str, float]:
        """Returns concept detection rate (CDR) over all videos in video_dir."""
        video_files = [
            f for f in os.listdir(self.video_dir)
            if f.endswith((".mp4", ".avi", ".mov"))
        ]

        if not video_files:
            print(f"No video files found in {self.video_dir}")
            return {"fire_detection_rate": 0.0}

        fire_count = 0
        for video_name in video_files:
            video_path = os.path.join(self.video_dir, video_name)
            if self.process_video(video_path):
                print("fire detected in", video_name)
                fire_count += 1

        cdr = fire_count / len(video_files)
        return {"fire_detection_rate": cdr, "videos_with_fire": fire_count, "total_videos": len(video_files)}


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
