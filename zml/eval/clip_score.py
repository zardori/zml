import argparse
import gc
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
NUM_FRAMES = 8  # evenly sampled from 49-frame videos


class VideoClipScorer:
    def __init__(
        self,
        video_dir: str,
        prompts: list[str],
        num_frames: int = NUM_FRAMES,
        device: str | None = None,
    ):
        self.video_dir = video_dir
        self.prompts = prompts
        self.num_frames = num_frames
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> tuple[CLIPModel, CLIPProcessor]:
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self.device)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        model.eval()
        return model, processor

    def _sample_frames(self, video_path: str) -> list[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            raise RuntimeError(f"Video has no frames: {video_path}")

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames

    def score_video(
        self,
        video_path: str,
        prompt: str,
        model: CLIPModel,
        processor: CLIPProcessor,
    ) -> float:
        frames = self._sample_frames(video_path)
        pil_frames = [Image.fromarray(f) for f in frames]

        pixel_values = processor(images=pil_frames, return_tensors="pt")["pixel_values"].to(self.device)
        text_inputs = processor(text=[prompt], return_tensors="pt", truncation=True, max_length=77).to(self.device)

        with torch.no_grad():
            # Transformers 5.x changed get_image/text_features return types; use sub-models directly
            image_pool = model.vision_model(pixel_values=pixel_values).pooler_output
            image_features = F.normalize(model.visual_projection(image_pool), dim=-1)  # (N, D)

            text_pool = model.text_model(**text_inputs).pooler_output
            text_features = F.normalize(model.text_projection(text_pool), dim=-1)      # (1, D)

        # Per-frame cosine similarity, averaged across frames
        per_frame_scores = (image_features @ text_features.T).squeeze(-1)  # (N,)
        return per_frame_scores.mean().item()

    def process_videos(self) -> list[float]:
        """Returns per-video CLIP scores for all videos in video_dir."""
        video_files = sorted(
            [f for f in os.listdir(self.video_dir) if f.endswith((".mp4", ".avi", ".mov"))],
            key=lambda f: int(f.split("_")[1].split(".")[0]),
        )

        if not video_files:
            print(f"No video files found in {self.video_dir}")
            return []

        n = min(len(video_files), len(self.prompts))
        if n < len(video_files):
            print(f"Warning: {len(video_files)} videos but {len(self.prompts)} prompts — scoring first {n}")

        model, processor = self._load_model()
        scores: list[float] = []
        try:
            for i in range(n):
                video_path = os.path.join(self.video_dir, video_files[i])
                score = self.score_video(video_path, self.prompts[i], model, processor)
                scores.append(score)
        finally:
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()

        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CLIP score for videos against their prompts")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--prompts_file", type=str, required=True, help="Text file with one prompt per line, matching video index order")
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES, help="Number of frames to sample per video")
    parser.add_argument("--device", type=str, default=None, help="Device to run CLIP on (default: cuda if available)")
    args = parser.parse_args()

    with open(args.prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]

    scorer = VideoClipScorer(
        video_dir=args.video_dir,
        prompts=prompts,
        num_frames=args.num_frames,
        device=args.device,
    )
    scores = scorer.process_videos()
    if scores:
        arr = np.array(scores)
        print({"scores": scores, "mean": float(arr.mean()), "std": float(arr.std())})
