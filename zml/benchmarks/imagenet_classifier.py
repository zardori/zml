"""Per-frame ImageNet classification with a pretrained ResNet-50.

This is the instrument behind the ESR/PSR object-erasure metric (``docs/imagenet_objects.md``). ESD's
object protocol — which T2VUnlearning states it follows, without naming its own classifier — scores
generated samples with a pretrained ImageNet ResNet-50, and VideoEraser reports ResNet-50 explicitly,
so that is what we use.

Frames come in as BGR uint8 (the format ``cv2.VideoCapture`` and ``decode_to_bgr_frames`` produce) and
are put through the weights' own inference transforms, so a video frame is preprocessed exactly like
an ImageNet validation image: resize to 232 on the short side, center crop 224, normalize.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torchvision.models import ResNet50_Weights, resnet50

from zml.benchmarks.imagenet_classes import IMAGENETTE_CLASSES

DEFAULT_TOP_K = 5
BATCH_SIZE = 32  # 49 frames at 224px is small; batching only keeps a long video off the GPU at once


def _assert_class_indices(categories: list[str]) -> None:
    """Guard the hardcoded index table against a torchvision that orders its categories differently.

    Every ESR/PSR number is read off one column of the output layer, so a silent reordering would
    corrupt all of them while still producing plausible-looking results.
    """
    wrong = {n: (i, categories[i]) for n, i in IMAGENETTE_CLASSES.items() if categories[i] != n}
    if wrong:
        raise RuntimeError(
            f"IMAGENETTE_CLASSES disagrees with this torchvision's category order: {wrong}. "
            "Fix zml/benchmarks/imagenet_classes.py before trusting any ESR/PSR number."
        )


@dataclass
class FrameClassification:
    """Per-frame classification of one clip, aligned to the input frame order."""

    topk_indices: np.ndarray  # (num_frames, k) ImageNet-1k indices, most probable first
    target_probs: np.ndarray  # (num_frames,) softmax probability of the requested target class


class ImageNetFrameClassifier:
    """Wraps a frozen ResNet-50; one instance can score any number of clips and target classes."""

    def __init__(self, device: str | None = None, batch_size: int = BATCH_SIZE):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        weights = ResNet50_Weights.IMAGENET1K_V2
        _assert_class_indices(weights.meta["categories"])
        self.transforms = weights.transforms()
        self.model = resnet50(weights=weights).eval().to(self.device)
        print(f"ImageNetFrameClassifier ready on {self.device} (ResNet-50 IMAGENET1K_V2)")

    def _probs(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Softmax probabilities for every frame, shape (num_frames, 1000)."""
        # BGR uint8 HWC -> RGB uint8 CHW, which is what the weights' transforms expect.
        batch = torch.from_numpy(np.stack([f[:, :, ::-1] for f in frames])).permute(0, 3, 1, 2)
        out = []
        with torch.no_grad():
            for start in range(0, len(batch), self.batch_size):
                chunk = self.transforms(batch[start : start + self.batch_size]).to(self.device)
                out.append(torch.softmax(self.model(chunk), dim=1).cpu())
        return torch.cat(out)

    def classify(
        self, frames: list[np.ndarray], target_index: int, k: int = DEFAULT_TOP_K
    ) -> FrameClassification:
        """Top-k predictions per frame plus the target class's probability per frame."""
        if not frames:
            return FrameClassification(
                topk_indices=np.empty((0, k), dtype=np.int64), target_probs=np.empty(0, dtype=np.float32)
            )
        probs = self._probs(frames)
        return FrameClassification(
            topk_indices=probs.topk(k, dim=1).indices.numpy(),
            target_probs=probs[:, target_index].numpy(),
        )
