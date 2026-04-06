from ultralytics import YOLO
import cv2
import os
import argparse


class VideoFireDetector:
    # Pretrained fire-detection YOLOv8 weights (fire / smoke classes)
    MODEL_ID = "keremberke/yolov8n-fire-detection"

    def __init__(self, video_dir: str, conf_threshold: float = 0.4):
        self.model = YOLO(self.MODEL_ID)
        self.video_dir = video_dir
        self.conf_threshold = conf_threshold
        print("VideoFireDetector has been setup")

    def process_video(self, video_path: str) -> bool:
        """Returns True if fire is detected in any frame of the video."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fire_detected = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            # The fire-detection model returns class 0 = fire, class 1 = smoke
            for result in results:
                if len(result.boxes) > 0:
                    fire_detected = True
                    break

            if fire_detected:
                break

        cap.release()
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
                fire_count += 1

        cdr = fire_count / len(video_files)
        return {"fire_detection_rate": cdr, "videos_with_fire": fire_count, "total_videos": len(video_files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for fire in videos using a YOLOv8 fire-detection model")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory where the videos are saved")
    parser.add_argument("--conf_threshold", type=float, default=0.4, help="Detection confidence threshold")
    args = parser.parse_args()

    detector = VideoFireDetector(video_dir=args.input_dir, conf_threshold=args.conf_threshold)
    scores = detector.process_videos()
    print(scores)
