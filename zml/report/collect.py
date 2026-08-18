"""Assembling one week's ``data.json``: what happened, and what a person then wrote about it.

The contract with the curator (``.claude/skills/weekly-report``) is the reason this module exists in
the shape it does. Everything here is *derived* and regenerated on every ``collect``; a small set of
fields is *authored* and must survive regeneration untouched, because they are the only part of the
deck a script cannot produce. ``CURATED_EXPERIMENT_FIELDS`` names them, and ``merge_curation`` is the
guarantee — the same principle as ``zml/metrics_file.update_metrics_json``: a second pass must never
silently discard the first pass's work.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from experiments_index import Experiment, discover  # via the path shim in zml/report/__init__.py

from zml.report import REPO_ROOT, artifacts, media, metrics as metric_defs
from zml.report.artifacts import Dataset, Run
from zml.report.window import Commit, NotesChange, Window, commits_in, notes_changes
WEEKLY_DIR = REPO_ROOT / "report" / "weekly"
DATA_NAME = "data.json"

# Authored by the curator, never by collect(). Regeneration merges fresh facts underneath these.
CURATED_EXPERIMENT_FIELDS = ("commentary", "include", "highlight", "media")
CURATED_TOP_FIELDS = ("narrative",)

# A `ready`/`active` experiment is work in flight; the deck reports it as a plan, not as a result.
PLANNED_STATUSES = ("ready", "active")

# Prose long enough to be a finding rather than a status tick. Tuned against a real week: the
# reference-rewrite residue lands at 1-3 lines, a genuine result section at 40+.
SUBSTANTIAL_NOTES_LINES = 8


@dataclass(frozen=True)
class ExperimentRecord:
    """One experiment's week: its registry metadata, what ran, and what was written about it."""

    exp_id: str
    name: str
    rel_dir: str
    thread: str | None
    concept: str
    method: str
    status: str
    takeaway: str
    notes_change: NotesChange | None
    commits: list[Commit]
    runs: list[Run]

    @property
    def has_local_results(self) -> bool:
        return any(run.has_results for run in self.runs)

    @property
    def substantial(self) -> bool:
        """Enough happened to be worth a card by default, before any curation."""
        if self.has_local_results or any(run.failed for run in self.runs):
            return True
        added = len(self.notes_change.added_lines) if self.notes_change else 0
        return added >= SUBSTANTIAL_NOTES_LINES or any(c.is_finding for c in self.commits)


# --- rendering the facts into plain JSON ---------------------------------------------------------

def _scores_row(scores: dict[str, dict], concept: str) -> dict:
    """One eval's scores as ``{prompt_set: {metric_label: formatted}}`` plus the column order."""
    present: set[str] = set()
    for group in scores.values():
        present |= metric_defs.measured_keys(group)

    columns = metric_defs.headline_metrics(concept, present)
    rows = {}
    for name in metric_defs.ordered_prompt_sets(scores):
        group = scores[name]
        measured = metric_defs.measured_keys(group)
        rows[name] = {m.label: (m.format(group[m.key]) if m.key in measured else None) for m in columns}
    return {
        "columns": [
            {"label": m.label, "lower_is_better": m.lower_is_better} for m in columns
        ],
        "prompt_sets": rows,
    }


def _trajectory(run: Run, concept: str) -> list[dict]:
    """Per-checkpoint series for the sparklines: ``[{step, {metric_label: value}}]``."""
    series = []
    for point in run.evals:
        present: set[str] = set()
        for group in point.scores.values():
            present |= metric_defs.measured_keys(group)
        columns = metric_defs.headline_metrics(concept, present)
        series.append({
            "step": point.step,
            "values": {
                name: {m.label: point.scores[name].get(m.key) for m in columns if m.key in present}
                for name in metric_defs.ordered_prompt_sets(point.scores)
            },
        })
    return series


def _standalone_scores(table: dict | None, columns: tuple[metric_defs.Metric, ...]) -> dict | None:
    """``esr_psr.json`` / ``id_similarity.json`` rendered like an eval table.

    Both files key their headline on the erased class/identity, which is the row a reader wants.
    """
    if not isinstance(table, dict):
        return None
    per_erased = table.get("per_erased_class") or table.get("per_erased_identity")
    if not isinstance(per_erased, dict) or not per_erased:
        return None
    return {
        "columns": [{"label": m.label, "lower_is_better": m.lower_is_better} for m in columns],
        "prompt_sets": {
            target: {m.label: m.format(values.get(m.key)) for m in columns}
            for target, values in per_erased.items()
            if isinstance(values, dict)
        },
    }


def _dataset_json(dataset: Dataset) -> dict:
    return {
        "built": dataset.built,
        "skipped": dataset.skipped,
        "attempted": dataset.attempted,
        "screened": dataset.screened,
        "human_kept": dataset.human_kept,
        "usable": dataset.usable,
        # The yield is the result a dataset build reports; both concepts transferred so far were
        # blocked by it, so it belongs on the card as a headline, not in a footnote.
        "yield": None if dataset.usable is None else round(dataset.usable / max(dataset.built, 1), 3),
    }


def _run_json(run: Run, concept: str) -> dict:
    final = run.evals[-1].scores if run.evals else {}
    return {
        "dir": run.rel_dir,
        "arm": run.arm,
        "job_type": run.job_type,
        "cluster": run.cluster,
        "job_id": run.job_id,
        "outcome": run.outcome,
        "elapsed_s": run.elapsed_s,
        "git_sha": run.git_sha,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "ended_at": run.ended_at.isoformat() if run.ended_at else None,
        "steps_evaluated": [point.step for point in run.evals],
        "final_step": run.evals[-1].step if run.evals else None,
        "final_scores": _scores_row(final, concept) if final else None,
        "trajectory": _trajectory(run, concept),
        "esr_psr": _standalone_scores(run.esr_psr, metric_defs.ESR_PSR_METRICS),
        "id_similarity": _standalone_scores(run.id_similarity, metric_defs.ID_SIMILARITY_METRICS),
        "dataset": _dataset_json(run.dataset) if run.dataset else None,
        "health_notes": run.health_notes,
        "config": run.config,
    }


def _commit_json(commit: Commit) -> dict:
    return {
        "sha": commit.sha,
        "author": commit.author,
        "date": commit.date,
        "subject": commit.subject,
        "finding": commit.is_finding,
    }


def _experiment_json(record: ExperimentRecord) -> dict:
    notes = record.notes_change
    return {
        "id": record.exp_id,
        "name": record.name,
        "dir": record.rel_dir,
        "thread": record.thread,
        "concept": record.concept,
        "method": record.method,
        "status": record.status,
        "takeaway": record.takeaway,
        "planned": record.status in PLANNED_STATUSES,
        "notes": {
            "path": notes.rel_path if notes else f"{record.rel_dir}/notes.md",
            "is_new": bool(notes and notes.is_new),
            "added_lines": notes.added_lines if notes else [],
        },
        "commits": [_commit_json(c) for c in record.commits],
        "runs": [_run_json(run, record.concept) for run in record.runs],
        # Curated below by merge_curation; seeded here so a first render is not empty.
        "include": record.substantial,
        "highlight": False,
        "commentary": "",
        "media": [],
    }


# --- collection --------------------------------------------------------------------------------

def _commits_by_experiment(commits: list[Commit]) -> dict[str, list[Commit]]:
    by_exp: dict[str, list[Commit]] = {}
    for commit in commits:
        for exp_id in commit.exp_ids:
            by_exp.setdefault(exp_id, []).append(commit)
    return by_exp


def _runs_in_window(exp: Experiment, window: Window) -> list[Run]:
    """This experiment's output directories that overlap the window.

    The start side falls back to the directory-name timestamp because a run that never wrote a
    ``run_info.json`` (anything predating ``slurm/run_info.sh``) still has one.
    """
    exp_dir = REPO_ROOT / "experiments" / exp.rel_dir
    return [
        run
        for run in artifacts.discover_runs(exp.exp_id, exp_dir, REPO_ROOT)
        if window.overlaps(run.started_at or run.dir_time, run.when)
    ]


def _records(window: Window) -> tuple[list[ExperimentRecord], list[Commit]]:
    """Every experiment the window touched, whether via git or via a run on disk.

    Both sources are needed and neither subsumes the other: a job can finish without anyone updating
    its notes (the thing a weekly review exists to catch), and notes can be written about a run whose
    outputs live on someone else's machine.
    """
    experiments, _ = discover()
    commits = commits_in(window)
    changes = notes_changes(window)
    by_exp = _commits_by_experiment(commits)

    records = [
        ExperimentRecord(
            exp_id=exp.exp_id,
            name=exp.name,
            rel_dir=f"experiments/{exp.rel_dir.as_posix()}",
            thread=exp.thread,
            concept=exp.concept,
            method=exp.method,
            status=exp.status,
            takeaway=exp.takeaway,
            notes_change=changes.get(exp.exp_id),
            commits=by_exp.get(exp.exp_id, []),
            runs=runs,
        )
        for exp in sorted(experiments, key=lambda e: e.exp_id)
        if (runs := _runs_in_window(exp, window)) or exp.exp_id in changes or exp.exp_id in by_exp
    ]
    return records, commits


def collect(window: Window) -> dict:
    """Everything that happened in ``window``, as the JSON the renderer and the curator both read."""
    records, commits = _records(window)
    return {
        "week": window.label,
        "range": {"start": window.start.isoformat(), "end": window.end.isoformat()},
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "narrative": {"headline": "", "summary": "", "next_week": []},
        "experiments": [_experiment_json(record) for record in records],
        "gaps": _gaps(records),
        "commits": [_commit_json(commit) for commit in commits],
    }


def _has_results_outside_window(record: ExperimentRecord) -> bool:
    """Whether any output dir of this experiment carries results, ignoring the window.

    ``record.runs`` holds only the runs the window overlaps, so a baseline measured weeks ago and
    still quoted this week would otherwise be reported as un-pulled.
    """
    exp_dir = REPO_ROOT / record.rel_dir
    if not exp_dir.is_dir():
        return False
    return any(run.has_results for run in artifacts.discover_runs(record.exp_id, exp_dir, REPO_ROOT))


def _gaps(records: list[ExperimentRecord]) -> list[dict]:
    """Experiments the week clearly touched but whose results are not on this machine.

    ``pull_results.sh`` never downloads ``notes.md``, so notes describing a run always arrive by git
    while the numbers behind them arrive only if someone pulled. Omitting these would make the deck
    quietly wrong; listing them makes the next action obvious.
    """
    gaps = []
    for record in records:
        if record.has_local_results or record.status in PLANNED_STATUSES:
            continue
        if not record.notes_change:
            continue
        # Results pulled for an earlier week are still on this machine; only the *window* filter
        # hid them. Flagging those would tell the reader to re-pull data they already have, and
        # would print "not on this machine" next to numbers the deck quotes from disk.
        if _has_results_outside_window(record):
            continue
        number = "".join(ch for ch in record.exp_id if ch.isdigit()).lstrip("0")
        gaps.append({
            "id": record.exp_id,
            "name": record.name,
            "thread": record.thread,
            "method": record.method,
            "fix": f"./pull_results.sh --range {number}",
        })
    return gaps


# --- curation-preserving merge -------------------------------------------------------------------

def merge_curation(fresh: dict, existing: dict | None) -> dict:
    """Re-apply the authored fields of ``existing`` on top of freshly collected facts.

    Experiments are matched by id, so a re-collect that picks up new runs, drops an experiment or
    reorders the week keeps every piece of writing attached to the right card.
    """
    if not existing:
        return fresh

    merged = dict(fresh)
    for key in CURATED_TOP_FIELDS:
        if existing.get(key):
            merged[key] = existing[key]

    previous = {exp["id"]: exp for exp in existing.get("experiments", []) if "id" in exp}
    for experiment in merged["experiments"]:
        old = previous.get(experiment["id"])
        if not old:
            continue
        for key in CURATED_EXPERIMENT_FIELDS:
            if key in old:
                experiment[key] = old[key]
    return merged


# --- default media --------------------------------------------------------------------------------

def _media_stem(experiment: dict, run: dict, candidate: media.Candidate) -> str:
    parts = [experiment["id"], run["arm"] or "", candidate.path.stem]
    return "_".join(part for part in parts if part).replace(".", "_")


def _candidates_for(run: dict) -> list[media.Candidate]:
    """What this run could show, preferring the dataset pairing when it built one.

    A precompute run's original/edited pair *is* its result — the deck should show the training
    target next to what the model produced, not a checkpoint's eval.
    """
    output_dir = REPO_ROOT / run["dir"]
    if run["dataset"]:
        return media.dataset_candidates(output_dir)
    return media.eval_candidates(output_dir, run["final_step"])


def seed_media(data: dict, frames: int, max_clips: int) -> list[str]:
    """Fill in a default strip (and a few clips) for cards that have no curated media yet.

    Only ever *adds*, and only where ``media`` is empty — an experiment the curator emptied on
    purpose stays empty, which is what makes re-collecting safe to do at any point.
    """
    warnings: list[str] = []
    media_dir = week_dir(data["week"]) / media.MEDIA_DIR_NAME

    # Clips are the scarce resource: highlighted cards first, then the newest work — experiment
    # numbers are chronological, so the highest ids are this week's runs.
    candidates = [
        exp for exp in data["experiments"] if exp["include"] and not exp["planned"] and exp["runs"]
    ]
    ordered = sorted(
        sorted(candidates, key=lambda exp: exp["id"], reverse=True),
        key=lambda exp: not exp["highlight"],
    )
    clips_left = max_clips

    for experiment in ordered:
        if experiment["media"]:
            continue
        run = experiment["runs"][-1]
        candidates = _candidates_for(run)[:2]  # a comparison is two clips; more is a gallery
        entries = []
        for candidate in candidates:
            stem = _media_stem(experiment, run, candidate)
            paths = media.extract_strip(candidate.path, media_dir, stem, frames)
            if not paths:
                warnings.append(f"{experiment['id']}: could not read frames from {candidate.path.name}")
                continue
            entry = {"kind": "strip", "label": candidate.label, "caption": "", "frames": paths}
            if clips_left > 0:
                src = media.stage_clip(candidate.path, media_dir, stem)
                if src:
                    entry["clip"] = src
                    clips_left -= 1
            entries.append(entry)
        experiment["media"] = entries

    return warnings


def week_dir(label: str) -> Path:
    return WEEKLY_DIR / label


def load_data(label: str) -> dict | None:
    path = week_dir(label) / DATA_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_data(label: str, data: dict) -> Path:
    path = week_dir(label) / DATA_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return path
