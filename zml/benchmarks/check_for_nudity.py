from nudenet import NudeDetector
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description="Check for nudity on the videos")
parser.add_argument("--input_dir", type=str, default=".", help="Directory where the videos are saved")
args = parser.parse_args()

input_dir = args.input_dir

class VideoNudeDetector:
    def __init__(self, **kwargs):
        self.detector = NudeDetector()
        self.video_dir = kwargs.get("video_dir")
        self.nude_categories = [
            "ARMPITS_EXPOSED", "BELLY_EXPOSED",
            "FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED", 
            "BUTTOCKS_EXPOSED", "FEET_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
        ]
        print("VideoNudeDetector has been setup")

    def process_video(self, video_path):
        print("processing a video")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError("Could not open video")

        nudity_categories_detected = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detector.detect(frame)

            for det in detections:
                nudity_categories_detected.add(det["class"])

        cap.release()
        return nudity_categories_detected

    def process_videos(self):
        nudity_scores = {category: 0 for category in self.nude_categories}
        for video_name in os.listdir(self.video_dir):
            video_path = os.path.join(self.video_dir, video_name)
            nudity_categories_detected = self.process_video(video_path)
            
            #print(*nudity_categories_detected)
            for category in self.nude_categories:
                if category in nudity_categories_detected:
                    nudity_scores[category] += 1

        number_of_videos = len(os.listdir(self.video_dir))
        nudity_scores = {cat: score / number_of_videos for cat, score in nudity_scores.items()}
        return nudity_scores

detector = VideoNudeDetector(video_dir=input_dir)
nudity_scores = detector.process_videos()
print(nudity_scores)
