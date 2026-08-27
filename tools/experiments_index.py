#!/usr/bin/env python3
"""Generate ``experiments/INDEX.md`` from each experiment's ``notes.md`` frontmatter.

The experiment folders are the source of truth; this only renders them. Every experiment
carries a small YAML frontmatter block at the top of its ``notes.md``::

    ---
    status: superseded        # ready | active | done | superseded | abandoned
    concept: fire             # fire | nudity | imagenet | none
    method: frame_replace     # mirrors config.yaml `method`
    thread: frame_replace_fire  # required once archived
    takeaway: >
      eta=2 erases but motion collapses; superseded by exp057.
    ---

``--check`` validates without writing and exits non-zero on a problem. Its load-bearing
check is that no *live* config references a path under ``experiments/archive/`` -- that is
the invariant that keeps the archive genuinely dead rather than a hidden dependency.

Field reference and archive policy: ``docs/experiment_registry.md``.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
ARCHIVE_DIR = EXPERIMENTS_DIR / "archive"
OUTPUT_MD = EXPERIMENTS_DIR / "INDEX.md"

EXP_DIR_RE = re.compile(r"^exp\d{3}_")
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\s*\n", re.DOTALL)
FRONTMATTER_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:")

VALID_STATUSES = ("ready", "active", "done", "superseded", "abandoned")
VALID_CONCEPTS = ("fire", "nudity", "imagenet", "face", "none")
LIVE_STATUSES = ("ready", "active")  # work not yet finished — never archive one
RETIRED_STATUSES = ("superseded", "abandoned")

REQUIRED_FIELDS = ("status", "concept", "method", "takeaway")

# Every thread, in narrative order, each with the write-up that summarises it. Threads whose
# work is still live have no archive/ folder yet; they are listed so the taxonomy is one list.
THREAD_DOCS: dict[str, tuple[str, str | None]] = {
    "esd_fire": ("ESD on fire — the first erasure baseline", None),
    "unhype": ("UnHype hypernetwork port — abandoned", "docs/unhype_video_attempts.md"),
    "frame_replace_fire": ("frame_replace developed and validated on fire", "docs/frame_replace.md"),
    "baselines": ("Base-model reference evals (fire era)", None),
    "misc": ("Smoke tests and dead-end probes", None),
    "shared": ("Datasets and anchors shared across threads", None),
    "nudity": ("Transfer to nudity via split-prompt", "docs/split_prompt.md"),
    "imagenet": ("Transfer to ImageNet object classes", "docs/imagenet_objects.md"),
    "face_identity": ("Transfer to celebrity face/identity erasure", "docs/face_identity.md"),
}

NOT_RECORDED = "(not recorded)"
# Written by slurm/run_info.sh for every job, including ones that crash or hit their wall clock.
RUN_INFO_NAME = "run_info.json"


@dataclass(frozen=True)
class RunSummary:
    """Where an experiment's jobs ran and how long they took, rolled up across its runs."""

    cluster: str
    elapsed_s: int | None
    outcome: str
    count: int  # a grid contributes one run_info.json per combination

    def render(self) -> str:
        parts = [self.cluster or "?", format_elapsed(self.elapsed_s)]
        if self.outcome != "completed":
            parts.append(self.outcome)
        if self.count > 1:
            parts.append(f"×{self.count}")
        return " ".join(parts)


@dataclass(frozen=True)
class Experiment:
    """One experiment folder plus the metadata parsed from its notes."""

    exp_id: str  # "exp041"
    name: str  # "exp041_preservation_precompute"
    rel_dir: Path  # relative to experiments/, e.g. archive/unhype/exp016_unhype_fire
    status: str
    concept: str
    method: str
    thread: str | None
    takeaway: str
    run: RunSummary | None  # None for experiments that predate run_info.json, or never ran

    @property
    def archived(self) -> bool:
        return self.rel_dir.parts[0] == "archive"


def format_elapsed(seconds: int | None) -> str:
    """Wall time as `1h45m` / `12m`, the granularity you actually size an sbatch --time at."""
    if seconds is None:
        return "—"
    hours, minutes = divmod(round(seconds / 60), 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def collect_run(exp_dir: Path) -> RunSummary | None:
    """Roll up every run_info.json under one experiment.

    A grid writes one per combination, so the longest is what a resubmission has to survive —
    hence max elapsed rather than the newest run's. The outcome reported is the worst one, so a
    single timed-out arm cannot hide behind nine that finished.
    """
    records = []
    for path in sorted(exp_dir.glob(f"**/{RUN_INFO_NAME}")):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    if not records:
        return None

    elapsed = [r["elapsed_s"] for r in records if isinstance(r.get("elapsed_s"), int)]
    outcomes = {str(r.get("outcome", "?")) for r in records}
    for worst in ("running", "timeout", "failed", "completed"):
        if worst in outcomes:
            break
    return RunSummary(
        cluster=str(records[-1].get("cluster") or "?"),
        elapsed_s=max(elapsed) if elapsed else None,
        outcome=worst,
        count=len(records),
    )


def parse_frontmatter(notes: Path) -> tuple[dict, list[str]]:
    """Return the parsed frontmatter mapping and any problems found reading it."""
    if not notes.exists():
        return {}, [f"{notes.relative_to(REPO_ROOT)}: missing"]

    match = FRONTMATTER_RE.match(notes.read_text())
    if match is None:
        return {}, [f"{notes.relative_to(REPO_ROOT)}: no YAML frontmatter block"]

    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        return {}, [f"{notes.relative_to(REPO_ROOT)}: unparseable frontmatter ({exc})"]

    if not isinstance(data, dict):
        return {}, [f"{notes.relative_to(REPO_ROOT)}: frontmatter is not a mapping"]
    return data, []


def _yaml_scalar(value: str) -> str:
    """One-line YAML rendering, quoted only where a plain scalar would not read back unchanged."""
    return yaml.safe_dump(value, default_flow_style=True, width=2**16).split("\n", 1)[0]


def set_frontmatter_fields(text: str, updates: dict[str, str]) -> str:
    """Return ``text`` with these top-level frontmatter fields set; everything else is untouched.

    A field already present is rewritten where it stands (its old value's continuation lines — a
    folded ``takeaway: >`` block, say — go with it); a new one is appended to the block. This edits
    the text rather than round-tripping the mapping through ``yaml.dump`` so the diff stays on the
    lines that actually changed: a ``notes.md`` is read by people far more often than by tools, and
    its hand-written frontmatter should survive a machine setting one field.
    """
    match = FRONTMATTER_RE.match(text)
    if match is None:
        raise ValueError("no YAML frontmatter block")

    pending = dict(updates)
    kept: list[str] = []
    lines = match.group(1).split("\n")
    i = 0
    while i < len(lines):
        line, i = lines[i], i + 1
        key = FRONTMATTER_KEY_RE.match(line)
        if key and key.group(1) in pending:
            kept.append(f"{key.group(1)}: {_yaml_scalar(pending.pop(key.group(1)))}")
            while i < len(lines) and (not lines[i].strip() or lines[i][:1].isspace()):
                i += 1
        else:
            kept.append(line)
    kept += [f"{key}: {_yaml_scalar(value)}" for key, value in pending.items()]
    return text[: match.start(1)] + "\n".join(kept) + text[match.end(1) :]


def mark_submitted(notes: Path, cluster: str, job_ids: list[str],
                   when: datetime | None = None) -> str:
    """Record in an experiment's frontmatter that its jobs are now queued; return the stamp written.

    ``status: active`` — in flight is what the experiment now is, whatever it was before — plus a
    ``submitted`` line naming the cluster and the job ids, which is what ``status`` alone cannot
    say: when it went out, where, and what to look for in ``squeue``. Called by ``submit_job.py``
    once sbatch has accepted the jobs, so that nobody reading the registry afterwards — the weekly
    report, ``INDEX.md``, or the research agent, which is handed each experiment's status and
    takeaway and little else — takes a running experiment for one that was never submitted.
    """
    stamp = f"{(when or datetime.now()).strftime('%Y-%m-%d %H:%M')} {cluster}"
    if job_ids:
        stamp += f" job{'s' if len(job_ids) > 1 else ''} {','.join(job_ids)}"

    text = notes.read_text()
    updated = set_frontmatter_fields(text, {"status": "active", "submitted": stamp})
    if updated != text:
        notes.write_text(updated)
    return stamp


def validate(rel_dir: Path, data: dict) -> list[str]:
    where = f"experiments/{rel_dir}/notes.md"
    problems = [f"{where}: missing field '{f}'" for f in REQUIRED_FIELDS if not data.get(f)]

    status, concept = data.get("status"), data.get("concept")
    if status and status not in VALID_STATUSES:
        problems.append(f"{where}: unknown status '{status}' (expected one of {list(VALID_STATUSES)})")
    if concept and concept not in VALID_CONCEPTS:
        problems.append(f"{where}: unknown concept '{concept}' (expected one of {list(VALID_CONCEPTS)})")

    archived = rel_dir.parts[0] == "archive"
    thread = data.get("thread")
    if thread is not None and thread not in THREAD_DOCS:
        problems.append(f"{where}: unknown thread '{thread}' (add it to THREAD_DOCS first)")
    # A finished-and-retired run is legitimately `done`; only a live status is contradictory.
    if archived and status in LIVE_STATUSES:
        problems.append(f"{where}: archived but status is '{status}'")
    return problems


def discover() -> tuple[list[Experiment], list[str]]:
    """Collect every experiment folder (any depth) and the problems found parsing them.

    An experiment is a directory named ``expNNN_*`` holding a ``notes.md``. Keyed on the notes and
    not on ``config.yaml``, because the registry frontmatter lives in the notes and some experiments
    have no config at all — exp087 re-edited an existing dataset with a local tool and never
    submitted a job, so a config-keyed glob left it out of ``INDEX.md``, out of validation and out of
    reach of the archive tool, silently. Grid ``run_NNN/`` directories carry a config but no notes,
    so they drop out here rather than needing the name check to catch them.
    """
    experiments: list[Experiment] = []
    problems: list[str] = []

    for notes in sorted(EXPERIMENTS_DIR.glob("**/notes.md")):
        exp_dir = notes.parent
        if not EXP_DIR_RE.match(exp_dir.name):
            continue

        rel_dir = exp_dir.relative_to(EXPERIMENTS_DIR)
        data, read_problems = parse_frontmatter(exp_dir / "notes.md")
        problems += read_problems + (validate(rel_dir, data) if data else [])

        experiments.append(
            Experiment(
                exp_id=exp_dir.name.split("_")[0],
                name=exp_dir.name,
                rel_dir=rel_dir,
                status=data.get("status", "?"),
                concept=data.get("concept", "?"),
                method=data.get("method", "?"),
                thread=data.get("thread"),
                takeaway=" ".join(str(data.get("takeaway", NOT_RECORDED)).split()),
                run=collect_run(exp_dir),
            )
        )
    return experiments, problems


def find_misfiled_experiments(experiments: list[Experiment]) -> list[str]:
    """An experiment must sit under its own thread: ``<thread>/`` live, ``archive/<thread>/`` retired.

    Kept out of ``validate()`` — and so out of ``discover()`` — on purpose. This is a property of
    where the folder *is*, not of whether its notes parse, and ``regroup_experiments.py`` has to be
    able to run on a tree that is failing it. A check that blocked its own remedy would be useless.
    """
    problems = []
    for exp in experiments:
        expected = exp.rel_dir.parts[1:-1] if exp.archived else exp.rel_dir.parts[:-1]
        if expected == (exp.thread,):
            continue
        where = f"experiments/{exp.rel_dir}/notes.md"
        if not expected:
            root = "experiments/archive/" if exp.archived else "experiments/"
            problems.append(
                f"{where}: sits directly in {root} — run tools/regroup_experiments.py to file it "
                f"under experiments/{exp.thread or '<thread>'}/"
            )
        else:
            problems.append(
                f"{where}: thread '{exp.thread}' does not match its folder '{'/'.join(expected)}'"
            )
    return problems


def find_duplicate_numbers(experiments: list[Experiment]) -> list[str]:
    """One number, one experiment — across every thread, live and archived alike.

    Numbering is contested three ways (two people and an autonomous agent), and threads are
    separate folders, so two experiments can take the same number without any single `ls` showing
    it. That happened with exp137, claimed by both the imagenet and nudity threads two days apart,
    and nothing caught it: numbers are how notes, commits and the weekly deck refer to runs, so a
    collision silently makes half those references ambiguous. `git fetch` before picking a number.
    """
    by_number: dict[str, list[Experiment]] = {}
    for exp in experiments:
        by_number.setdefault(exp.exp_id, []).append(exp)
    return [
        f"{exp_id} is claimed by {len(dupes)} experiments: "
        + ", ".join(f"experiments/{e.rel_dir}" for e in sorted(dupes, key=lambda e: e.rel_dir))
        + " — renumber the one claimed later (git log --diff-filter=A on each folder)"
        for exp_id, dupes in sorted(by_number.items())
        if len(dupes) > 1
    ]


def discover_remote_experiment_dirs(ref: str) -> dict[str, str]:
    """Map exp_id -> "<thread-path>/expNNN_name" for every experiment that exists at a git ref.

    `discover()` only sees the local working tree, so it cannot catch a number a teammate has
    already pushed but this branch has not pulled yet -- exactly how exp137 and exp147 collided.
    Returns {} (rather than raising) if the ref cannot be resolved, e.g. no network or a stale
    fetch: the caller treats that as "nothing to cross-check", not as a validation failure.
    """
    try:
        listing = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", ref, "--", "experiments"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        return {}

    remote: dict[str, str] = {}
    for line in listing.splitlines():
        if not line.endswith("/notes.md"):
            continue
        rel_dir = Path(line).parent.relative_to("experiments")
        if not EXP_DIR_RE.match(rel_dir.name):
            continue
        remote[rel_dir.name.split("_")[0]] = str(rel_dir)
    return remote


def find_remote_conflicts(experiments: list[Experiment], ref: str) -> list[str]:
    """Flag a number that resolves to a *different* folder locally than at `ref`.

    Complements `find_duplicate_numbers`, which only sees what one working tree has checked out.
    """
    remote = discover_remote_experiment_dirs(ref)
    if not remote:
        return []
    return [
        f"{exp.exp_id} conflicts with {ref}: local experiments/{exp.rel_dir} vs "
        f"{ref}:experiments/{remote[exp.exp_id]} — git fetch and renumber before committing"
        for exp in experiments
        if exp.exp_id in remote and remote[exp.exp_id] != str(exp.rel_dir)
    ]


def find_archive_references(experiments: list[Experiment]) -> list[str]:
    """Live configs must never point into ``experiments/archive/``.

    A live run depending on archived data means the archive is not dead, and the next
    person to prune it breaks a running experiment.
    """
    problems = []
    for exp in experiments:
        if exp.archived:
            continue
        config = EXPERIMENTS_DIR / exp.rel_dir / "config.yaml"
        if not config.exists():
            continue  # a tool-driven experiment with no submitted job, e.g. exp087
        for lineno, line in enumerate(config.read_text().splitlines(), start=1):
            if "experiments/archive/" in line and not line.lstrip().startswith("#"):
                problems.append(
                    f"experiments/{exp.rel_dir}/config.yaml:{lineno}: live config references "
                    f"archived data — un-archive that experiment or copy the data forward"
                )
    return problems


def render_table(experiments: list[Experiment], *, link_prefix: str = "") -> list[str]:
    lines = [
        "| ID | Concept | Method | Status | Run | Takeaway |",
        "|----|---------|--------|--------|-----|----------|",
    ]
    for exp in experiments:
        link = f"[{exp.exp_id}]({link_prefix}{exp.rel_dir}/notes.md)"
        run = exp.run.render() if exp.run else "—"
        lines.append(
            f"| {link} | {exp.concept} | {exp.method} | {exp.status} | {run} | {exp.takeaway} |"
        )
    return lines


def render_index(experiments: list[Experiment]) -> str:
    active = [e for e in experiments if not e.archived]
    archived = [e for e in experiments if e.archived]

    lines = [
        "# Experiment index",
        "",
        "<!-- Generated by tools/experiments_index.py — do not edit by hand. -->",
        "<!-- Edit the frontmatter of the experiment's notes.md and regenerate. -->",
        "",
        "Policy, field reference and how to retire an experiment: **`docs/experiment_registry.md`**.",
        "",
        f"## Active ({len(active)})",
        "",
        "Everything the current work depends on. A run here is either in flight, or a result /",
        "dataset that a live config still reads. Grouped by thread, because experiment *numbers*",
        "interleave across the three concepts being unlearned in parallel — `exp064` and `exp119`",
        "are the same thread, `exp065` and `exp066` are not neighbours in anything but numbering.",
        "",
    ]

    for thread, (blurb, doc) in THREAD_DOCS.items():
        in_thread = [e for e in active if e.thread == thread]
        if not in_thread:
            continue
        lines += ["", f"### `{thread}` ({len(in_thread)})", "", blurb + "."]
        if doc:
            lines.append(f"Full write-up: **`{doc}`**.")
        lines.append("")
        lines += render_table(in_thread)

    retirable = [e for e in active if e.status in RETIRED_STATUSES]
    if retirable:
        lines += [
            "",
            "> Ready to archive (`superseded`/`abandoned`, still live): "
            + ", ".join(e.exp_id for e in retirable)
            + " — run `tools/archive_experiment.py`.",
        ]

    lines += ["", f"## Archive ({len(archived)})", "",
              "Retired threads. Kept for provenance; nothing live reads them.", ""]

    for thread, (blurb, doc) in THREAD_DOCS.items():
        in_thread = [e for e in archived if e.rel_dir.parts[1] == thread]
        if not in_thread:
            continue
        lines += ["", f"### `{thread}` ({len(in_thread)})", "", blurb + "."]
        if doc:
            lines.append(f"Full write-up: **`{doc}`**.")
        lines.append("")
        lines += render_table(in_thread)

    unknown = [e for e in archived if e.rel_dir.parts[1] not in THREAD_DOCS]
    if unknown:
        lines += ["", "### Unclassified", "",
                  "These sit under `archive/` in a thread this tool does not know about:", ""]
        lines += render_table(unknown)

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate frontmatter and the archive invariant without writing; non-zero exit on failure.",
    )
    parser.add_argument(
        "--remote-ref",
        default=None,
        help="Also cross-check experiment numbers against this git ref (e.g. origin/master) to "
             "catch a collision with work not yet pulled. Fetch it yourself first; this only reads.",
    )
    args = parser.parse_args()

    experiments, problems = discover()
    problems += (find_misfiled_experiments(experiments) + find_duplicate_numbers(experiments)
                 + find_archive_references(experiments))
    if args.remote_ref:
        problems += find_remote_conflicts(experiments, args.remote_ref)

    if problems:
        print(f"{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        if args.check:
            return 1

    if args.check:
        active = sum(1 for e in experiments if not e.archived)
        print(f"OK — {len(experiments)} experiments ({active} active, {len(experiments) - active} archived)")
        return 0

    OUTPUT_MD.write_text(render_index(experiments))
    print(f"Wrote {OUTPUT_MD.relative_to(REPO_ROOT)} ({len(experiments)} experiments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
