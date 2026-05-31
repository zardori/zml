from __future__ import annotations

import argparse
import gc
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download

try:
    from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
    from dover.models import DOVER
    DOVER_AVAILABLE = True
except Exception:
    DOVER_AVAILABLE = False

DOVER_REPO_ID = "teowu/DOVER"
DOVER_WEIGHTS_FILE = "DOVER.pth"
DOVER_CONFIG_URL = "https://raw.githubusercontent.com/QualityAssessment/DOVER/master/dover.yml"
DOVER_CACHE_DIR = Path.home() / ".cache" / "dover"

# ImageNet normalization used by DOVER preprocessing
_MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
_STD = torch.FloatTensor([58.395, 57.12, 57.375])

# Per-branch normalization params derived from DOVER training distribution.
# Raw logits are standardized with these, then mapped through sigmoid to [0, 1].
# Source: fuse_results() in the official evaluate_one_video.py
_TECHNICAL_MEAN, _TECHNICAL_STD = 0.1107, 0.07355
_AESTHETIC_MEAN, _AESTHETIC_STD = -0.08285, 0.03774


def _normalize_score(raw: float, mean: float, std: float) -> float:
    """Standardize a raw DOVER logit and apply sigmoid to get a [0, 1] quality score."""
    return float(1 / (1 + np.exp(-((raw - mean) / std))))


def _load_dover_config() -> dict:
    """Download dover.yml from GitHub on first use and cache it locally."""
    config_path = DOVER_CACHE_DIR / "dover.yml"
    if not config_path.exists():
        DOVER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DOVER_CONFIG_URL, config_path)
    with open(config_path) as f:
        return yaml.safe_load(f)


class VideoDoverScorer:
    def __init__(self, video_dir: str, device: str | None = None):
        self.video_dir = video_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> tuple[DOVER, dict]:
        opt = _load_dover_config()
        weights_path = hf_hub_download(repo_id=DOVER_REPO_ID, filename=DOVER_WEIGHTS_FILE)
        model = DOVER(**opt["model"]["args"]).to(self.device)
        model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=False)
        )
        model.eval()
        return model, opt["data"]["val-l1080p"]["args"]

    def score_video(self, video_path: str, model: DOVER, data_args: dict) -> dict[str, float]:
        temporal_samplers = {}
        for stype, sopt in data_args["sample_types"].items():
            if "t_frag" not in sopt:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

        views, _ = spatial_temporal_view_decomposition(
            video_path, data_args["sample_types"], temporal_samplers
        )

        for k, v in views.items():
            num_clips = data_args["sample_types"][k].get("num_clips", 1)
            views[k] = (
                ((v.permute(1, 2, 3, 0) - _MEAN) / _STD)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to(self.device)
            )

        with torch.no_grad():
            # results[0] = fragments branch (technical), results[1] = resize branch (aesthetic)
            results = [r.mean().item() for r in model(views)]

        return {
            "technical": _normalize_score(results[0], _TECHNICAL_MEAN, _TECHNICAL_STD),
            "aesthetic": _normalize_score(results[1], _AESTHETIC_MEAN, _AESTHETIC_STD),
        }

    def process_videos(self) -> dict[str, list[float]]:
        """Returns per-video technical and aesthetic quality scores for all videos in video_dir."""
        video_files = [
            f for f in os.listdir(self.video_dir)
            if f.endswith((".mp4", ".avi", ".mov"))
        ]

        if not video_files:
            print(f"No video files found in {self.video_dir}")
            return {"technical": [], "aesthetic": []}

        model, data_args = self._load_model()
        technical: list[float] = []
        aesthetic: list[float] = []

        try:
            for video_name in video_files:
                video_path = os.path.join(self.video_dir, video_name)
                scores = self.score_video(video_path, model, data_args)
                technical.append(scores["technical"])
                aesthetic.append(scores["aesthetic"])
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

        return {"technical": technical, "aesthetic": aesthetic}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute DOVER quality scores for videos")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    scorer = VideoDoverScorer(video_dir=args.video_dir, device=args.device)
    scores = scorer.process_videos()
    if scores["technical"]:
        tech = np.array(scores["technical"])
        aes = np.array(scores["aesthetic"])
        print({
            "technical_scores": scores["technical"],
            "technical_mean": float(tech.mean()),
            "technical_std": float(tech.std()),
            "aesthetic_scores": scores["aesthetic"],
            "aesthetic_mean": float(aes.mean()),
            "aesthetic_std": float(aes.std()),
        })
