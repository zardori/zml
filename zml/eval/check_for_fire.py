from ultralytics import YOLO
import cv2
import os
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_THRESHOLD = 0.8

class VideoFireDetector:
    # Pretrained fire-detection YOLOv8 weights (fire / smoke classes)
    MODEL_ID = "SalahALHaismawi/yolov26-fire-detection"
    # class 0 = fire, class 1 = smoke, class 2 = other
    CLASS_NAMES = {0: "fire", 1: "smoke", 2: "other"}
    FIRE_CLASS = 0

    def __init__(self, video_dir: str, conf_threshold: float = DEFAULT_THRESHOLD, debug: bool = False, debug_dir: str | None = None):
        model_path = hf_hub_download(
            repo_id=self.MODEL_ID,
            filename="best.pt"
        )
        self.model = YOLO(model_path)
        self.video_dir = video_dir
        self.conf_threshold = conf_threshold
        self.debug = debug
        if debug_dir is None:
            video_dir_path = Path(video_dir).resolve()
            debug_dir = str(video_dir_path.parent / f"{video_dir_path.name}_debug")
        self.debug_dir = debug_dir
        print("VideoFireDetector has been setup")

    def process_video(self, video_path: str) -> bool:
        """Returns True if fire is detected in any frame of the video.

        In debug mode, writes an annotated video with bounding boxes and confidence
        scores to self.debug_dir.
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

        fire_detected = False
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.debug:
                # Run with lower threshold and fire-only to see candidate detections
                debug_threshold = 0.5
                results = self.model(frame, conf=debug_threshold, classes=[self.FIRE_CLASS], verbose=False)
                for result in results:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf >= self.conf_threshold:
                            fire_detected = True
                        print(f"  frame {frame_idx}: fire conf={conf:.3f}{' (above threshold)' if conf >= self.conf_threshold else ''}")
                annotated = results[0].plot()
                writer.write(annotated)
            else:
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                for result in results:
                    for box in result.boxes:
                        if int(box.cls[0]) == self.FIRE_CLASS:
                            fire_detected = True
                            break
                    if fire_detected:
                        break
                if fire_detected:
                    break

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()
        return fire_detected

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
    parser.add_argument("--conf_threshold", type=float, default=DEFAULT_THRESHOLD, help="Detection confidence threshold")
    parser.add_argument("--debug", action="store_true", help="Write annotated debug videos with fire bounding boxes and confidence scores")
    parser.add_argument("--debug_dir", type=str, default=None, help="Directory for debug videos (default: <input_dir>_debug sibling)")
    args = parser.parse_args()

    detector = VideoFireDetector(video_dir=args.input_dir, conf_threshold=args.conf_threshold, debug=args.debug, debug_dir=args.debug_dir)
    scores = detector.process_videos()
    print(scores)
