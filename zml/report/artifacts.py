"""What an experiment left on disk: when its jobs ran, and what they scored.

Every result artifact here is gitignored (``experiments/**/outputs_*/``), so this module reads the
working tree and nothing else — a run that has not been pulled with ``./pull_results.sh`` is simply
absent, which is a fact the report has to state rather than hide (see ``collect.gaps``).

Timestamps are ranked deliberately. Filesystem mtimes are **not** used at any point: ``pull_results.sh``
rsyncs with ``-u`` so mtimes come from the cluster, and the post-hoc scorers (``tools/score_dover.py``
and friends) rewrite ``metrics.json`` days after the run, which would date a July run to this week.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from zml.results_io import EvalPoint, eval_trajectory

DIR_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Written by slurm/run_info.sh for every job type, including ones that crash or hit the wall clock —
# the only artifact a failed run is guaranteed to leave behind.
RUN_INFO_NAME = "run_info.json"
# Written by scripts/eval.py, on success only.
EVAL_RUNTIME_NAME = "runtime.json"

# Standalone eval entrypoints write their headline outside eval_step_*/: ImageNet ESR/PSR and face
# ID-similarity are whole-run numbers, not per-checkpoint ones.
ESR_PSR_NAME = "esr_psr.json"
ID_SIMILARITY_NAME = "id_similarity.json"

# Dataset builds (split_prompt / frame_replace_split precompute) report a yield, not a score.
METADATA_NAME = "metadata.json"
SKIPPED_NAME = "skipped.json"
SCREENED_SUFFIX = "_screened.json"

FAILED_OUTCOMES = ("timeout", "failed")


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _parse_dir_timestamp(name: str) -> datetime | None:
    """``outputs_20260815_014333`` / ``grid_20260806_211043`` -> local-time datetime.

    Stamped by ``submit_job.py`` from the submitter's wall clock with no zone, so it is read as local
    time. Present for every run of every job type, which is why it is the fallback of last resort.
    """
    _, _, stamp = name.partition("_")
    try:
        return datetime.strptime(stamp, DIR_TIMESTAMP_FORMAT).astimezone()
    except ValueError:
        return None


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.astimezone()


@dataclass(frozen=True)
class Dataset:
    """A precompute run's yield — the result a dataset build reports instead of a metric."""

    built: int
    skipped: int
    screened: int | None  # kept by tools/screen_split_dataset.py, when it has been run
    human_kept: int | None  # kept by human review (metadata_human_filtered*.json)

    @property
    def attempted(self) -> int:
        return self.built + self.skipped

    @property
    def usable(self) -> int | None:
        return self.human_kept if self.human_kept is not None else self.screened


@dataclass
class Run:
    """One job's output directory, with whatever it managed to record."""

    exp_id: str
    rel_dir: str  # relative to the repo root
    arm: str | None  # "run_002" for a grid arm, else None
    dir_time: datetime | None
    started_at: datetime | None
    ended_at: datetime | None
    cluster: str | None
    job_id: str | None
    job_type: str | None
    outcome: str | None
    elapsed_s: int | None
    git_sha: str | None
    evals: list[EvalPoint] = field(default_factory=list)
    train: dict[str, dict[str, float]] = field(default_factory=dict)
    health_notes: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    esr_psr: dict | None = None
    id_similarity: dict | None = None
    dataset: Dataset | None = None

    @property
    def when(self) -> datetime | None:
        """Best available "this run happened at" instant, most trustworthy source first."""
        return self.ended_at or self.started_at or self.dir_time

    @property
    def failed(self) -> bool:
        return self.outcome in FAILED_OUTCOMES

    @property
    def has_results(self) -> bool:
        return bool(self.evals or self.esr_psr or self.id_similarity or self.dataset)


def _read_run_info(output_dir: Path) -> dict:
    """The run_info.json for this output directory.

    A grid arm keeps its own; a plain run keeps one at the top. Globbing rather than joining covers
    both, and the newest wins when a directory was resubmitted into.
    """
    candidates = sorted(output_dir.glob(f"**/{RUN_INFO_NAME}"))
    for path in reversed(candidates):
        info = _load_json(path)
        if isinstance(info, dict):
            return info
    return {}


def _read_timings(output_dir: Path, info: dict) -> tuple[datetime | None, datetime | None]:
    """Start/end, falling back through the three artifacts that record them.

    ``summary.json``'s ``runtime`` is refreshed on every metrics flush, so it dates a job that was
    killed by the wall clock; ``runtime.json`` only exists for evals that finished.
    """
    started = _parse_iso(info.get("started_at"))
    ended = _parse_iso(info.get("ended_at"))

    runtime = (_load_json(output_dir / "summary.json") or {}).get("runtime", {})
    if isinstance(runtime, dict):
        started = started or _parse_iso(runtime.get("started_at"))
        ended = ended or _parse_iso(runtime.get("updated_at"))

    eval_runtime = _load_json(output_dir / EVAL_RUNTIME_NAME)
    if isinstance(eval_runtime, dict):
        started = started or _parse_iso(eval_runtime.get("started_at"))
        ended = ended or _parse_iso(eval_runtime.get("finished_at"))
    return started, ended


def _read_dataset(exp_dir: Path, output_dir: Path) -> Dataset | None:
    metadata = _load_json(output_dir / METADATA_NAME)
    if not isinstance(metadata, list):
        return None

    skipped = _load_json(output_dir / SKIPPED_NAME)
    # The screening and human-review artifacts are tracked and live beside the outputs directory,
    # named after it, because they survive the outputs being deleted.
    screened = _load_json(exp_dir / f"{output_dir.name}{SCREENED_SUFFIX}")
    human = [
        len(rows)
        for path in sorted(exp_dir.glob("metadata_human_filtered*.json"))
        if isinstance(rows := _load_json(path), list)
    ]
    return Dataset(
        built=len(metadata),
        skipped=len(skipped) if isinstance(skipped, list) else 0,
        screened=len(screened) if isinstance(screened, list) else None,
        human_kept=max(human) if human else None,
    )


def _read_summary(output_dir: Path) -> tuple[dict, dict, list[str]]:
    summary = _load_json(output_dir / "summary.json")
    if not isinstance(summary, dict):
        return {}, {}, []
    health = summary.get("health") or {}
    notes = health.get("notes") if isinstance(health, dict) else None
    return summary.get("config") or {}, summary.get("train") or {}, list(notes or [])


def _build_run(exp_id: str, exp_dir: Path, output_dir: Path, arm: str | None, repo_root: Path) -> Run:
    info = _read_run_info(output_dir)
    started, ended = _read_timings(output_dir, info)
    config, train, health_notes = _read_summary(output_dir)
    # A grid arm's own `outputs/` has no timestamp; it lives on the enclosing `grid_{TS}/`.
    stamped = output_dir.parent.parent if arm else output_dir

    return Run(
        exp_id=exp_id,
        rel_dir=output_dir.relative_to(repo_root).as_posix(),
        arm=arm,
        dir_time=_parse_dir_timestamp(stamped.name),
        started_at=started,
        ended_at=ended,
        cluster=info.get("cluster"),
        job_id=info.get("job_id"),
        job_type=info.get("job_type"),
        outcome=info.get("outcome"),
        elapsed_s=info.get("elapsed_s") if isinstance(info.get("elapsed_s"), int) else None,
        git_sha=info.get("git_sha"),
        evals=eval_trajectory(output_dir),
        train=train,
        health_notes=health_notes,
        config=config,
        esr_psr=_load_json(output_dir / ESR_PSR_NAME) or None,
        id_similarity=_load_json(output_dir / ID_SIMILARITY_NAME) or None,
        dataset=_read_dataset(exp_dir, output_dir),
    )


def discover_runs(exp_id: str, exp_dir: Path, repo_root: Path) -> list[Run]:
    """Every output directory under one experiment, oldest first.

    Two shapes exist: ``outputs_{TS}/`` for a single run, and ``grid_{TS}/run_NNN/outputs/`` for a
    grid — the grid arms carry the timestamp on the grid directory, not on their own outputs.
    """
    runs = [
        _build_run(exp_id, exp_dir, output_dir, None, repo_root)
        for output_dir in sorted(exp_dir.glob("outputs_*"))
        if output_dir.is_dir()
    ]
    runs += [
        _build_run(exp_id, exp_dir, arm / "outputs", arm.name, repo_root)
        for grid in sorted(exp_dir.glob("grid_*"))
        for arm in sorted(grid.glob("run_*"))
        if (arm / "outputs").is_dir()
    ]
    # Undatable runs sort last; datetime.min cannot be made tz-aware, so the flag carries the order.
    epoch = datetime.fromtimestamp(0).astimezone()
    return sorted(runs, key=lambda run: (run.when is None, run.when or epoch))
