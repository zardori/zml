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

import glob
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

from zml.benchmarks.check_for_nudity import VideoNudeDetector

# split_latent_frame=7 -> latent frames [7:] are the concept region. Under the 1+4k causal mapping
# latent frame 7 begins at pixel frame 1+4*(7-1)=25, so pixel frames >= 25 are the "second half".
DEFAULT_SECOND_HALF_START = 25
CLIP_TYPES = ("A", "B", "C", "combined")


@dataclass
class Config:
    # Directory of .mp4 clips (e.g. an exp059 outputs_*/videos dir). Mutually exclusive with grid_dir.
    videos_dir: str | None = None
    # A submit_job.py grid root (e.g. experiments/exp074_.../grid_TIMESTAMP): report every run_*/
    # subdir and aggregate by the varied config field named by grid_key (e.g. "split_step_frac").
    # Use this for hyperparameter sweeps instead of one report job per grid combination.
    grid_dir: str | None = None
    grid_key: str = "split_step_frac"
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


def _report_videos_dir(videos_dir: str, config: Config) -> dict:
    """Per-frame nudity report over one directory of clips. Pure function of (videos_dir, config)."""
    detector = VideoNudeDetector(video_dir=videos_dir)

    video_files = sorted(f for f in os.listdir(videos_dir) if f.endswith((".mp4", ".avi", ".mov")))
    if not video_files:
        raise ValueError(f"No videos in {videos_dir}")

    per_clip: list[dict] = []
    for name in video_files:
        frames = _read_frames(os.path.join(videos_dir, name))
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

    return {
        "videos_dir": videos_dir,
        "second_half_start": config.second_half_start,
        "frame_nudity_threshold": config.frame_nudity_threshold,
        "summary_by_clip_type": summary,
        "combined_verdict": verdict,
        "per_clip": per_clip,
    }


def _find_videos_dir(run_dir: str) -> str:
    """Grid runs write to run_dir/outputs/videos (no timestamp); tolerate a timestamped outputs_* too."""
    candidates = sorted(glob.glob(os.path.join(run_dir, "outputs*", "videos")))
    if not candidates:
        raise ValueError(f"No outputs*/videos under {run_dir}")
    return candidates[-1]


def _run_grid_key_value(run_dir: str, grid_key: str) -> float:
    with open(os.path.join(run_dir, "config.yaml")) as f:
        run_config = yaml.safe_load(f)
    if grid_key not in run_config:
        raise ValueError(f"{run_dir}/config.yaml has no field {grid_key!r}")
    return run_config[grid_key]


def _main_grid(config: Config) -> None:
    run_dirs = sorted(glob.glob(os.path.join(config.grid_dir, "run_*")))
    if not run_dirs:
        raise ValueError(f"No run_* subdirs under {config.grid_dir}")

    sweep: list[dict] = []
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        key_value = _run_grid_key_value(run_dir, config.grid_key)
        report = _report_videos_dir(_find_videos_dir(run_dir), config)
        with open(os.path.join(config.output_dir, f"nudity_report_{run_name}.json"), "w") as f:
            json.dump(report, f, indent=2)

        combined = report["summary_by_clip_type"].get("combined")
        verdict = report["combined_verdict"]
        sweep.append({
            "run": run_name,
            config.grid_key: key_value,
            "n": combined["n"] if combined else 0,
            "first_half_max": combined["mean_first_half_max"] if combined else None,
            "second_half_max": combined["mean_second_half_max"] if combined else None,
            "gap": verdict["combined_second_minus_first_half_max"] if verdict else None,
            "localized_to_second_half": verdict["localized_to_second_half"] if verdict else None,
        })

    sweep.sort(key=lambda r: r[config.grid_key])

    # A candidate optimum: the smallest split value that both localizes (second half clearly nude,
    # first half clean) and clears the detection threshold with margin. Ties/near-ties and seam
    # quality (visible-splice artifacts) are NOT captured by this per-frame nudity score and still
    # need a human look at the actual clips before picking a final value.
    candidates = [r for r in sweep if r["localized_to_second_half"] and r["first_half_max"] < config.frame_nudity_threshold]
    best = min(candidates, key=lambda r: r[config.grid_key]) if candidates else None

    out = {
        "grid_dir": config.grid_dir,
        "grid_key": config.grid_key,
        "frame_nudity_threshold": config.frame_nudity_threshold,
        "sweep": sweep,
        "suggested_best": best,
    }
    out_path = os.path.join(config.output_dir, "sweep_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"sweep": sweep, "suggested_best": best}, indent=2))
    print(f"Wrote {out_path} (+ one nudity_report_run_*.json per run)")


def main(config: Config) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    if config.grid_dir and config.videos_dir:
        raise ValueError("Set exactly one of videos_dir / grid_dir, not both")
    if config.grid_dir:
        _main_grid(config)
        return
    if not config.videos_dir:
        raise ValueError("Set one of videos_dir / grid_dir")

    report = _report_videos_dir(config.videos_dir, config)
    out_path = os.path.join(config.output_dir, "nudity_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({
        "summary_by_clip_type": report["summary_by_clip_type"],
        "combined_verdict": report["combined_verdict"],
    }, indent=2))
    print(f"Wrote {out_path}")
