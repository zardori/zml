"""Score VBench Subject Consistency on already-generated eval videos, into their metrics.json.

Why this is a separate, after-the-fact tool
-------------------------------------------
Subject Consistency is one of the two utility dimensions T2VUnlearning (arXiv:2505.17550) use as
their *entire* preservation evidence, so we report it to close the "you omitted their metrics"
objection (see ``docs/comparability_t2vunlearning.md`` §6). It needs DINO ViT-B/16 over every frame,
which is a second model in the eval loop for a number nobody reads during training — and like DOVER
it is computable from the saved ``.mp4`` files alone, on any machine, with no cluster job. So it runs
post-hoc, exactly like ``tools/score_dover.py`` and ``tools/score_nudity_frame_rate.py``.

**Report it only next to a motion column.** The metric scores similarity to the first frame and to
the previous frame, so a frozen clip approaches 100. Our best checkpoint *gains* on it (94.24 ->
96.41 on VBench's own 72 prompts) while losing 36% of its motion on the very same clips, and
T2VUnlearning's own method *loses* 0.83. Printed alone the number says we preserve capability better
than they do; printed beside motion it says the metric cannot see temporal collapse, which is the
finding. The scorer cannot enforce that, so it is written here.

Run:
    uv run python tools/score_subject_consistency.py experiments/exp106_.../run_002/outputs
    uv run python tools/score_subject_consistency.py <dir> --dry-run
"""

import argparse
import json
from pathlib import Path

import numpy as np

from zml.eval.subject_consistency import VideoSubjectConsistencyScorer
from zml.metrics_file import update_metrics_json
from zml.video_files import VIDEO_EXTENSIONS

# VBench reports this dimension on a 0-100 scale; the scorer returns a raw cosine mean in [0, 1].
# Stored scaled so the field is directly comparable with their published 95.53 / 94.70.
SCALE = 100.0


def _fields(scores: list[float]) -> dict[str, object]:
    values = np.array(scores) if scores else np.array([0.0])
    return {
        "subject_consistency_scores": [round(SCALE * v, 4) for v in scores],
        "subject_consistency_mean": round(float(SCALE * values.mean()), 4),
        "subject_consistency_std": round(float(SCALE * values.std()), 4),
    }


def score_output_dir(output_dir: Path, device: str | None, dry_run: bool) -> None:
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

        # Scored outside the lock, merged under it — see zml/metrics_file.py. Concurrent post-hoc
        # scorers over the same run used to lose whichever wrote first.
        pending: dict[str, dict] = {}
        for set_dir in sorted(p for p in step_dir.iterdir() if p.is_dir()):
            set_name = set_dir.name
            # Only prompt-set entries carry scores; `_`-prefixed keys are provenance metadata.
            if set_name not in metrics or not isinstance(metrics[set_name], dict):
                continue
            if not any(f.suffix in VIDEO_EXTENSIONS for f in set_dir.iterdir()):
                print(f"  {step_dir.name}/{set_name}: no videos, skipping")
                continue

            scores = VideoSubjectConsistencyScorer(video_dir=str(set_dir), device=device).process_videos()
            fields = _fields(scores)
            motion = metrics[set_name].get("motion_score_mean")
            print(
                f"  {step_dir.name}/{set_name}: subject_consistency "
                f"{fields['subject_consistency_mean']:.2f} over {len(scores)} videos "
                f"(motion {motion} — read them together)"
            )
            pending[set_name] = fields

        if pending and not dry_run:
            update_metrics_json(metrics_path, pending)
            print(f"  {step_dir.name}: metrics.json updated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output_dir", type=Path,
                        help="A run's outputs_{timestamp}/ or run_*/outputs/ dir with eval_step_*/")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    parser.add_argument("--dry-run", action="store_true", help="report scores without writing")
    args = parser.parse_args()

    score_output_dir(args.output_dir, args.device, args.dry_run)
