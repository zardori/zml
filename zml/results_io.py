"""Reading a finished run's evaluation scores off disk.

A run records the same eval twice, and *neither copy is complete*:

- ``summary.json``'s ``eval`` list — written live by ``zml/unlearn/metrics_log.py``, seven keys per
  prompt set, values kept to 4 significant figures. No post-hoc scorer ever touches it.
- ``eval_step_{STEP}/metrics.json`` — the full metric set including per-video arrays, and the file the
  post-hoc scorers (``tools/score_dover.py``, ``score_nudity_frame_rate.py``, ``score_q16.py``,
  ``score_subject_consistency.py``) merge their columns into. But ``zml/unlearn/eval.py`` rounds it to
  **2 decimal places** on write, which flattens every area score — ``0.003924`` lands as ``0.0``.

So reading a run's scores is a merge, not a file choice: the summary supplies precision for the keys
it has, the step file supplies everything added since. Getting this wrong is silent — you either lose
the area column to rounding or lose the DOVER/Q16 columns to reading the wrong file. Three callers
(``tools/build_results_table.py``, ``tools/build_frame_replace_table.py``,
``tools/weekly_report.py``) need it, so it lives here once.

Reading only. Merging *into* ``metrics.json`` belongs to ``zml/metrics_file.py``, which locks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# DOVER's import fails on helios (aarch64) and the scorer records a mean of exactly 0.0 rather than
# failing the job, so a zero here means "not measured" and must never be read as a quality score.
# It is also why the merge below lets a step file's DOVER value override the summary's: a run
# evaluated on helios and DOVER-scored locally afterwards has the real number only in the step file.
DOVER_KEYS = ("dover_technical_mean", "dover_aesthetic_mean")

Scores = dict[str, dict[str, float]]
"""``{prompt_set: {metric: value}}`` — prompt sets are ``concept``/``related``/``unrelated``/``anchor``."""


@dataclass(frozen=True)
class EvalPoint:
    """One checkpoint's merged scores, as recorded at training step ``step``."""

    step: int
    scores: Scores


def _step_number(path: Path) -> int:
    return int(path.name.rsplit("_", 1)[-1])


def _step_dirs(output_dir: Path) -> list[Path]:
    """``eval_step_*`` directories, ordered by step number rather than lexicographically."""
    return sorted(
        (p for p in output_dir.glob("eval_step_*") if p.name.rsplit("_", 1)[-1].isdigit()),
        key=_step_number,
    )


def _load_json(path: Path) -> dict:
    """Parsed JSON, or ``{}`` — a file truncated by a killed job must not take the caller down."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _summary_scores_by_step(output_dir: Path) -> dict[int, Scores]:
    entries = _load_json(output_dir / "summary.json").get("eval", [])
    return {
        int(entry["step"]): entry.get("scores", {})
        for entry in entries
        if entry.get("type") == "eval" and entry.get("step") is not None
    }


def _step_scores(step_dir: Path) -> Scores:
    """A step file's scores, minus the non-prompt-set bookkeeping keys ``evaluate`` writes alongside."""
    return {
        name: fields
        for name, fields in _load_json(step_dir / "metrics.json").items()
        if isinstance(fields, dict)  # drops "_negative_prompt" and any future scalar annotation
    }


def _merge_group(precise: dict[str, float], full: dict[str, float]) -> dict[str, float]:
    """One prompt set: ``full``'s coverage with ``precise``'s significant figures."""
    merged = dict(full)
    for key, value in precise.items():
        # The step file wins only where the summary's value is an unmeasured DOVER zero.
        if key in DOVER_KEYS and not value and full.get(key):
            continue
        merged[key] = value
    return merged


def _merge(precise: Scores, full: Scores) -> Scores:
    groups = list(full) + [name for name in precise if name not in full]
    return {name: _merge_group(precise.get(name, {}), full.get(name, {})) for name in groups}


def eval_trajectory(output_dir: Path) -> list[EvalPoint]:
    """Every checkpoint's merged scores for a run, ordered by step.

    Covers the three shapes on disk: both files present (merged), step files only (runs predating the
    metrics recorder, or evals recovered by ``tools/score_eval_videos.py``), and summary only (a run
    whose eval directories were not pulled — ``pull_results.sh --no-videos`` still brings the summary).
    """
    summary_scores = _summary_scores_by_step(output_dir)
    steps_on_disk = {_step_number(d): d for d in _step_dirs(output_dir)}

    return [
        EvalPoint(
            step,
            _merge(
                summary_scores.get(step, {}),
                _step_scores(steps_on_disk[step]) if step in steps_on_disk else {},
            ),
        )
        for step in sorted(set(summary_scores) | set(steps_on_disk))
    ]


def latest_eval_scores(output_dir: Path, step: int | None = None) -> Scores:
    """Scores for one checkpoint: ``step=None`` selects the final eval, else that exact step.

    Raises rather than returning ``{}`` when nothing is found — a table row silently rendering as all
    blank cells is the failure this replaces.
    """
    trajectory = [point for point in eval_trajectory(output_dir) if point.scores]
    if not trajectory:
        raise FileNotFoundError(f"No eval scores found under {output_dir}")

    if step is None:
        return trajectory[-1].scores
    for point in trajectory:
        if point.step == step:
            return point.scores
    raise ValueError(f"No eval at step {step} under {output_dir}")


def dover_score(group_scores: dict[str, float], key: str = "dover_aesthetic_mean") -> float | None:
    """A DOVER mean, or ``None`` when the run never measured it (helios writes a literal 0.0)."""
    value = group_scores.get(key)
    return value if value else None
