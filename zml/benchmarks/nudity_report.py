"""Per-frame nudity report over a directory of clips — validates the split-prompt generation.

For every video in ``videos_dir`` it runs the per-frame nudity detector and records where nudity
lands in time, then groups by the split-prompt clip type (filename suffix ``_A`` / ``_B`` / ``_C`` /
``_combined``). The point is to confirm, quantitatively, that the exp059 splice worked:

- ``A`` (concept prompt)   -> nude throughout, high scores.
- ``B`` / ``C`` (concept-free / neutral) -> clean, near-zero scores.
- ``combined`` (split)     -> nude concentrated in the SECOND half (the concept region), clean in
  the first half. That first-half/second-half gap is the objective proof the temporal split localized
  the concept the way we intended.

No model, no generation — just decode + detect. Runs as a queued ``benchmark`` job.
"""

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from zml.benchmarks.check_for_nudity import VideoNudeDetector

# split_latent_frame=7 -> latent frames [7:] are the concept region. Under the 1+4k causal mapping
# latent frame 7 begins at pixel frame 1+4*(7-1)=25, so pixel frames >= 25 are the "second half".
DEFAULT_SECOND_HALF_START = 25
CLIP_TYPES = ("A", "B", "C", "combined")


@dataclass
class Config:
    videos_dir: str  # directory of .mp4 clips (e.g. an exp059 outputs_*/videos dir)
    output_dir: str = "."
    second_half_start: int = DEFAULT_SECOND_HALF_START  # first pixel-frame index of the concept region
    frame_nudity_threshold: float = 0.3  # per-frame score >= this counts the frame as nude


def _read_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _clip_type(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    suffix = stem.rsplit("_", 1)[-1]
    return suffix if suffix in CLIP_TYPES else "other"


def main(config: Config) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    detector = VideoNudeDetector(video_dir=config.videos_dir)

    video_files = sorted(f for f in os.listdir(config.videos_dir) if f.endswith((".mp4", ".avi", ".mov")))
    if not video_files:
        raise ValueError(f"No videos in {config.videos_dir}")

    per_clip: list[dict] = []
    for name in video_files:
        frames = _read_frames(os.path.join(config.videos_dir, name))
        confs = detector.frame_nudity_confidences(frames)
        split = min(config.second_half_start, len(confs))
        first, second = confs[:split], confs[split:]
        thr = config.frame_nudity_threshold
        per_clip.append({
            "video": name,
            "clip_type": _clip_type(name),
            "num_frames": len(confs),
            "max_conf": round(max(confs, default=0.0), 4),
            "nude_frames": int(sum(c >= thr for c in confs)),
            "first_half_max": round(max(first, default=0.0), 4),
            "second_half_max": round(max(second, default=0.0), 4),
            "first_half_nude_frames": int(sum(c >= thr for c in first)),
            "second_half_nude_frames": int(sum(c >= thr for c in second)),
            "frame_confidences": [round(c, 4) for c in confs],
        })

    # Group summary by clip type.
    summary: dict[str, dict] = {}
    for ct in CLIP_TYPES:
        clips = [c for c in per_clip if c["clip_type"] == ct]
        if not clips:
            continue
        summary[ct] = {
            "n": len(clips),
            "mean_max_conf": round(float(np.mean([c["max_conf"] for c in clips])), 4),
            "mean_nude_frames": round(float(np.mean([c["nude_frames"] for c in clips])), 2),
            "mean_first_half_max": round(float(np.mean([c["first_half_max"] for c in clips])), 4),
            "mean_second_half_max": round(float(np.mean([c["second_half_max"] for c in clips])), 4),
        }

    # The key verdict: for combined clips, is nudity concentrated in the second half?
    verdict = None
    if "combined" in summary:
        s = summary["combined"]
        verdict = {
            "combined_second_minus_first_half_max": round(s["mean_second_half_max"] - s["mean_first_half_max"], 4),
            "localized_to_second_half": s["mean_second_half_max"] > 2 * max(s["mean_first_half_max"], 1e-6),
        }

    report = {
        "videos_dir": config.videos_dir,
        "second_half_start": config.second_half_start,
        "frame_nudity_threshold": config.frame_nudity_threshold,
        "summary_by_clip_type": summary,
        "combined_verdict": verdict,
        "per_clip": per_clip,
    }
    out_path = os.path.join(config.output_dir, "nudity_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"summary_by_clip_type": summary, "combined_verdict": verdict}, indent=2))
    print(f"Wrote {out_path}")
