"""Score DOVER on already-generated eval videos and merge the results into their metrics.json.

Why this is a separate, after-the-fact tool: DOVER is the one metric in this pipeline that actually
measures technical video quality (the thing human review kept catching and
`clip_score`/`colorfulness`/`motion` kept missing — see exp073), but every run that has needed it
was run on **helios**, whose compute nodes are aarch64 (GH200). `pyproject.toml` gates `dover` and
`decord` on ``platform_machine == 'x86_64'``, so on helios the import fails, ``DOVER_AVAILABLE`` is
False, and the eval writes ``0.0`` into the DOVER fields. Those zeros are a placeholder for "not
measured", not a quality score, and they have been misread as data before.

The fix does not need a cluster job at all: DOVER scoring is post-hoc over saved ``.mp4`` files, so
it runs anywhere DOVER imports — an x86_64 login node, athena, or a laptop. Pull a run's eval videos
with ``pull_results.sh`` (they are not excluded by default) and point this at the output dir. It
rewrites only the DOVER fields of each ``eval_step_*/metrics.json``, leaving every other score
untouched, so `tools/build_*_table.py` (which already treats a DOVER mean of 0.0 as missing) starts
reporting real numbers with no further changes.

Run:
    uv run python tools/score_dover.py experiments/nudity/exp080_.../outputs_20260806_120000
    uv run python tools/score_dover.py <dir> --dry-run     # report what would change
"""

import argparse
import json
from pathlib import Path

import numpy as np

from zml.eval.dover_scorer import DOVER_AVAILABLE, VideoDoverScorer
from zml.metrics_file import update_metrics_json

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov")


def _score_set(video_dir: Path, device: str | None) -> dict[str, list[float]]:
    return VideoDoverScorer(video_dir=str(video_dir), device=device).process_videos()


def _dover_fields(scores: dict[str, list[float]]) -> dict[str, object]:
    technical = np.array(scores["technical"]) if scores["technical"] else np.array([0.0])
    aesthetic = np.array(scores["aesthetic"]) if scores["aesthetic"] else np.array([0.0])
    return {
        "dover_technical_scores": [round(v, 4) for v in scores["technical"]],
        "dover_technical_mean": round(float(technical.mean()), 4),
        "dover_technical_std": round(float(technical.std()), 4),
        "dover_aesthetic_scores": [round(v, 4) for v in scores["aesthetic"]],
        "dover_aesthetic_mean": round(float(aesthetic.mean()), 4),
        "dover_aesthetic_std": round(float(aesthetic.std()), 4),
    }


def score_output_dir(output_dir: Path, device: str | None, dry_run: bool,
                     sets: tuple[str, ...] | None = None) -> None:
    step_dirs = sorted(
        output_dir.glob("eval_step_*"),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )
    if not step_dirs:
        raise FileNotFoundError(f"No eval_step_*/ directories under {output_dir}")

    for step_dir in step_dirs:
        metrics_path = step_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"  {step_dir.name}: no metrics.json, skipping")
            continue
        metrics = json.loads(metrics_path.read_text())

        # Scored outside the lock, merged under it — see zml/metrics_file.py. A concurrent
        # frame-rate pass over the same run used to lose whichever tool wrote first.
        pending: dict[str, dict] = {}
        for set_dir in sorted(p for p in step_dir.iterdir() if p.is_dir()):
            set_name = set_dir.name
            # A run evaluates concept + related + unrelated, but checkpoint selection only reads
            # `concept`. Scoring all three triples the wall clock for numbers nobody ranks on, so
            # --sets lets a long backlog be narrowed to what a decision actually needs.
            if sets and set_name not in sets:
                continue
            # Only prompt-set entries carry scores; `_`-prefixed keys are provenance metadata.
            if set_name not in metrics or not isinstance(metrics[set_name], dict):
                continue
            if not any(f.suffix in VIDEO_SUFFIXES for f in set_dir.iterdir()):
                print(f"  {step_dir.name}/{set_name}: no videos, skipping")
                continue

            scores = _score_set(set_dir, device)
            fields = _dover_fields(scores)
            print(
                f"  {step_dir.name}/{set_name}: "
                f"technical {metrics[set_name].get('dover_technical_mean')} -> "
                f"{fields['dover_technical_mean']}, "
                f"aesthetic {metrics[set_name].get('dover_aesthetic_mean')} -> "
                f"{fields['dover_aesthetic_mean']}"
            )
            pending[set_name] = fields

        if pending and not dry_run:
            update_metrics_json(metrics_path, pending)
            print(f"  {step_dir.name}: metrics.json updated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path,
                        help="A run's outputs_{timestamp}/ dir containing eval_step_*/ subdirs")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    parser.add_argument("--sets", nargs="+", default=None, metavar="NAME",
                        help="Only score these prompt sets (e.g. --sets concept). Default: all.")
    args = parser.parse_args()

    if not DOVER_AVAILABLE:
        raise SystemExit(
            "DOVER is not importable here. It is gated in pyproject.toml on x86_64 linux, so it is "
            "unavailable on helios (aarch64 GH200 compute nodes). Run this on an x86_64 machine "
            "(athena, an x86_64 login node, or locally) against pulled eval videos."
        )
    score_output_dir(args.output_dir, args.device, args.dry_run,
                     tuple(args.sets) if args.sets else None)
