"""Score an eval run's already-generated videos into its metrics.json, without regenerating.

An eval job does expensive GPU generation first and cheap CPU scoring last, so a job that runs out
of wall-clock in the scoring phase leaves a *complete* set of videos on disk and no metrics — the
whole cost is paid and none of the result is kept. That is exactly what happened to exp082/exp083
(all 4 grid runs generated every clip, then hit the SLURM time limit during NudeNet scoring). This
tool finishes those runs from the videos alone.

It reads the run's own ``config.yaml`` to find which prompt CSV produced each prompt-set directory
(so CLIP score pairs each clip with the prompt it was generated from) and which detector to score
with, then writes the same ``metrics.json`` the eval job would have written, via the same
``score_video_dir`` the eval path itself uses — so a recovered run is not scored differently from a
normal one.

Needs no GPU cluster job. Note DOVER only contributes on x86_64 (see CLAUDE.md); on a machine
without it the DOVER fields stay 0.0 and can be filled in later with ``tools/score_dover.py``.

Run:
    uv run python tools/score_eval_videos.py experiments/exp082_.../grid_.../run_001
    uv run python tools/score_eval_videos.py <run_dir> --dry-run
"""

import argparse
import json
from pathlib import Path

import yaml

from zml.unlearn.eval import _round_metrics, load_eval_prompts, score_video_dir

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov")
# Prompt-set directory name -> the config field naming the CSV it was generated from.
SET_TO_CONFIG_FIELD = {
    "concept": "control_concept_prompts",
    "related": "control_related_prompts",
    "unrelated": "control_unrelated_prompts",
}


def _output_dirs(run_dir: Path) -> list[Path]:
    """Every outputs dir under a run, across both experiment layouts.

    A grid run writes ``run_NNN/outputs/``; a single-run experiment writes
    ``expNNN_name/outputs_{timestamp}/`` (see the layout in CLAUDE.md). Supporting only the first
    made single-run experiments unrecoverable by this tool — which is how exp063's 115 base-model
    clips sat scored-never-written for a week despite the videos being on disk.
    """
    if (run_dir / "outputs").is_dir():
        return [run_dir / "outputs"]
    return sorted(p for p in run_dir.glob("outputs_*") if p.is_dir())


def score_run(run_dir: Path, dry_run: bool) -> None:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    concept = config.get("concept", "fire")
    concept_target = config.get("concept_target")
    negative_prompt = config.get("negative_prompt")

    step_dirs = sorted(
        (step for out in _output_dirs(run_dir) for step in out.glob("eval_step_*")),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )
    if not step_dirs:
        raise FileNotFoundError(
            f"No eval_step_*/ directories under {run_dir}/outputs/ or {run_dir}/outputs_*/"
        )

    for step_dir in step_dirs:
        metrics: dict[str, object] = {}
        for set_dir in sorted(p for p in step_dir.iterdir() if p.is_dir()):
            set_name = set_dir.name
            field = SET_TO_CONFIG_FIELD.get(set_name)
            if field is None or not config.get(field):
                print(f"  {step_dir.name}/{set_name}: no prompt CSV in config, skipping")
                continue
            videos = sorted(f for f in set_dir.iterdir() if f.suffix in VIDEO_SUFFIXES)
            if not videos:
                print(f"  {step_dir.name}/{set_name}: no videos, skipping")
                continue

            prompts = [ep.prompt for ep in load_eval_prompts(config[field])]
            if len(videos) > len(prompts):
                raise ValueError(
                    f"{set_dir} has {len(videos)} videos but {config[field]} only has "
                    f"{len(prompts)} prompts — wrong CSV for this run, refusing to mispair them."
                )
            # Truncate rather than require equality: a run with eval_num_prompts set, or one killed
            # partway through generating this set, has fewer clips than the CSV has rows. Clips are
            # written video_0..video_N-1 in prompt order and every scorer sorts, so the first
            # len(videos) prompts are the ones that produced them.
            if len(videos) < len(prompts):
                print(f"  {step_dir.name}/{set_name}: {len(videos)} videos < {len(prompts)} prompts, "
                      f"scoring the first {len(videos)}")
                prompts = prompts[:len(videos)]

            metrics[set_name] = score_video_dir(str(set_dir), prompts, concept, concept_target)
            scores = metrics[set_name]
            print(f"  {step_dir.name}/{set_name}: n={len(videos)} "
                  f"{concept}_detection_rate={scores[f'{concept}_detection_rate']:.3f} "
                  f"clip={scores['clip_score_mean']:.4f} "
                  f"colorfulness={scores['colorfulness_mean']:.2f} "
                  f"motion={scores['motion_score_mean']:.3f}")

        if not metrics:
            continue
        on_disk = dict(_round_metrics(metrics))
        if negative_prompt is not None:
            on_disk["_negative_prompt"] = negative_prompt
        if dry_run:
            print(f"  {step_dir.name}: would write metrics.json ({len(metrics)} prompt sets)")
        else:
            (step_dir / "metrics.json").write_text(json.dumps(on_disk, indent=2))
            print(f"  {step_dir.name}: metrics.json written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="+",
                        help="Run dir(s) holding config.yaml and outputs/eval_step_*/")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args()
    for run_dir in args.run_dir:
        print(f"== {run_dir} ==")
        score_run(run_dir, args.dry_run)
