"""VBench's Subject Consistency, reimplemented so we report the comparison paper's own metric.

Why we report a metric we distrust
----------------------------------
T2VUnlearning (arXiv:2505.17550) backs its "preserves generation capability" claim with exactly two
VBench dimensions, Object Class and Subject Consistency, and reports them only for HunyuanVideo
(Original 95.53 -> theirs 94.70 on this one). We report DOVER, motion, CLIP and colorfulness
instead. Reporting only our own instruments — on which our method looks worse — invites the
reasonable objection that theirs were omitted deliberately. So we report both.

**Subject Consistency rewards stillness.** It measures how similar a frame looks to the first frame
and to its predecessor, so a frozen clip scores near 1.0. Our current checkpoint costs ~-88% motion,
which this metric would read as a *strength*. That is the point: publishing it next to our motion
and DOVER columns is what turns "their metrics cannot see temporal collapse" from an excuse into a
demonstrated claim. Never report it alone — see docs/comparability_t2vunlearning.md.

The metric
----------
Faithful to VBench's `subject_consistency.py`: DINO ViT-B/16 CLS features per frame, L2-normalised,
then for every frame i > 0

    score_i = ( max(0, cos(f_0, f_i)) + max(0, cos(f_{i-1}, f_i)) ) / 2

and the video score is the mean over i. DINO rather than CLIP is VBench's choice and matters — DINO
features are sensitive to instance identity rather than semantic category, which is what "is it the
same subject throughout" needs.

Preprocessing is DINO's standard eval transform (resize 256 bicubic, center-crop 224, ImageNet
normalisation), again matching VBench. Weights come from HuggingFace `facebook/dino-vitb16`, which
is the same checkpoint as the `torch.hub` route VBench uses, without the hub's network fragility.

VBench's own protocol scores this over its 72 `subject_consistency` prompts. The metric itself is
prompt-agnostic, so it also runs on any clips we already have — useful for showing directly what it
does to a frozen video — but a number comparable with their 94.70 must come from those 72 prompts.
"""

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import ViTModel

from zml.video_files import list_video_files

DINO_MODEL_ID = "facebook/dino-vitb16"

# DINO's standard evaluation transform, as used by VBench.
RESIZE_SIZE = 256
CROP_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Frames per forward pass. 32 x 224px ViT-B/16 fits comfortably in 4 GB.
DEFAULT_BATCH_SIZE = 32


def _build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class VideoSubjectConsistencyScorer:
    """VBench Subject Consistency per video, over every clip in ``video_dir``.

    Scores lie in [0, 1] and are reported by VBench as percentages; multiply by 100 to compare
    against their 94.70. A single-frame or unreadable video scores 0.0.
    """

    def __init__(
        self,
        video_dir: str,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.video_dir = video_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.transform = _build_transform()
        self.model = ViTModel.from_pretrained(DINO_MODEL_ID).to(self.device).eval()

    @torch.no_grad()
    def _frame_features(self, frames_bgr: list[np.ndarray]) -> torch.Tensor:
        """L2-normalised DINO CLS features, one row per frame."""
        tensors = [
            self.transform(Image.fromarray(frame[:, :, ::-1])) for frame in frames_bgr
        ]
        features = []
        for start in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[start:start + self.batch_size]).to(self.device)
            cls = self.model(pixel_values=batch).last_hidden_state[:, 0]
            features.append(F.normalize(cls, dim=-1, p=2))
        return torch.cat(features)

    def score_video(self, video_path: str) -> float:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        if len(frames) < 2:
            return 0.0

        features = self._frame_features(frames)
        # Clamped at 0 exactly as VBench does: a negative cosine would otherwise let one bad frame
        # pull the mean below what "no consistency at all" should score.
        to_first = F.cosine_similarity(features[0].unsqueeze(0), features[1:]).clamp(min=0.0)
        to_previous = F.cosine_similarity(features[:-1], features[1:]).clamp(min=0.0)
        return float(((to_first + to_previous) / 2).mean())

    def process_videos(self) -> list[float]:
        return [
            self.score_video(f"{self.video_dir}/{name}")
            for name in list_video_files(self.video_dir)
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VBench Subject Consistency over a video directory")
    parser.add_argument("--input_dir", default=".")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    scores = VideoSubjectConsistencyScorer(video_dir=args.input_dir, device=args.device).process_videos()
    if scores:
        print(f"subject_consistency mean {100 * float(np.mean(scores)):.2f} over {len(scores)} videos")
