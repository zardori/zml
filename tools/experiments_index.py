#!/usr/bin/env python3
"""Generate ``experiments/INDEX.md`` from each experiment's ``notes.md`` frontmatter.

The experiment folders are the source of truth; this only renders them. Every experiment
carries a small YAML frontmatter block at the top of its ``notes.md``::

    ---
    status: superseded        # active | done | superseded | abandoned
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
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
ARCHIVE_DIR = EXPERIMENTS_DIR / "archive"
OUTPUT_MD = EXPERIMENTS_DIR / "INDEX.md"

EXP_DIR_RE = re.compile(r"^exp\d{3}_")
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\s*\n", re.DOTALL)

VALID_STATUSES = ("active", "done", "superseded", "abandoned")
VALID_CONCEPTS = ("fire", "nudity", "imagenet", "none")
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
}

NOT_RECORDED = "(not recorded)"


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

    @property
    def archived(self) -> bool:
        return self.rel_dir.parts[0] == "archive"


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
    if archived:
        if thread != rel_dir.parts[1]:
            problems.append(f"{where}: thread '{thread}' does not match its archive folder '{rel_dir.parts[1]}'")
        # A finished-and-retired run is legitimately `done`; only `active` is contradictory.
        if status == "active":
            problems.append(f"{where}: archived but status is 'active'")
    return problems


def discover() -> tuple[list[Experiment], list[str]]:
    """Collect every experiment folder (any depth) and the problems found parsing them.

    An experiment is a directory named ``expNNN_*`` holding a ``config.yaml``; the glob
    therefore also picks up grid ``run_NNN/`` configs, which the name check drops.
    """
    experiments: list[Experiment] = []
    problems: list[str] = []

    for config in sorted(EXPERIMENTS_DIR.glob("**/config.yaml")):
        exp_dir = config.parent
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
            )
        )
    return experiments, problems


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
        for lineno, line in enumerate(config.read_text().splitlines(), start=1):
            if "experiments/archive/" in line and not line.lstrip().startswith("#"):
                problems.append(
                    f"experiments/{exp.rel_dir}/config.yaml:{lineno}: live config references "
                    f"archived data — un-archive that experiment or copy the data forward"
                )
    return problems


def render_table(experiments: list[Experiment], *, link_prefix: str = "") -> list[str]:
    lines = [
        "| ID | Concept | Method | Status | Takeaway |",
        "|----|---------|--------|--------|----------|",
    ]
    for exp in experiments:
        link = f"[{exp.exp_id}]({link_prefix}{exp.rel_dir}/notes.md)"
        lines.append(f"| {link} | {exp.concept} | {exp.method} | {exp.status} | {exp.takeaway} |")
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
        "dataset that a live config still reads.",
        "",
    ]
    lines += render_table(active)

    retirable = [e for e in active if e.status in RETIRED_STATUSES]
    if retirable:
        lines += [
            "",
            "> Ready to archive (`superseded`/`abandoned`, still flat): "
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
    args = parser.parse_args()

    experiments, problems = discover()
    problems += find_archive_references(experiments)

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
