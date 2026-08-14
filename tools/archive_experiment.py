#!/usr/bin/env python3
"""Retire an experiment from ``experiments/<thread>/`` into ``experiments/archive/<thread>/``.

This is the *policy* half of retiring: which experiments may move, and what would break if they
did. The mechanics -- rewriting the references, carrying the untracked ``outputs_*``/``logs_*``
across, and stacking the move onto ``tools/migrate_experiments.sh`` for the other members --
live in ``experiment_moves.py`` and are shared with ``regroup_experiments.py``.

    tools/archive_experiment.py expNNN [expNNN ...]           # dry run, prints what it would do
    tools/archive_experiment.py expNNN [expNNN ...] --apply

The thread comes from each experiment's ``notes.md`` frontmatter (``thread:``), so the folder it
lands in and the group it renders under in ``INDEX.md`` can never disagree. Archiving is
therefore a move *sideways* -- ``experiments/imagenet/expNNN`` to
``experiments/archive/imagenet/expNNN`` -- and only the live/retired axis changes.

Refuses to move an experiment that is still live (``status`` ``ready`` or ``active``), or one
that any live config still references -- that reference would keep the archive alive as a
hidden dependency.

Policy and the cluster half of the migration: ``docs/experiment_registry.md``.
"""
from __future__ import annotations

import argparse
import sys

from experiment_moves import (
    Move,
    apply_moves,
    find_references,
    print_migration_footer,
    reference_files,
    report_plan,
)
from experiments_index import EXPERIMENTS_DIR, LIVE_STATUSES, discover, Experiment

ARCHIVE_ROOT = "archive"


def build_move(exp: Experiment) -> Move:
    """Retire in place under ``archive/``, keeping the thread grouping the live tree already has."""
    return Move(
        name=exp.name,
        thread=exp.thread,
        old_rel=f"experiments/{exp.rel_dir}",
        new_rel=f"experiments/{ARCHIVE_ROOT}/{exp.thread}/{exp.name}",
    )


def resolve(exp_ids: list[str], experiments: list[Experiment]) -> tuple[list[Experiment], list[str]]:
    by_id = {e.exp_id: e for e in experiments}
    selected, problems = [], []
    for exp_id in exp_ids:
        exp = by_id.get(exp_id)
        if exp is None:
            problems.append(f"{exp_id}: no such experiment")
        elif exp.archived:
            print(f"  = {exp.name} is already archived, skipping")
        elif exp.status in LIVE_STATUSES:
            problems.append(f"{exp_id}: status is '{exp.status}' — mark it superseded/abandoned/done first")
        elif not exp.thread:
            problems.append(f"{exp_id}: no 'thread' in its notes.md frontmatter")
        else:
            selected.append(exp)
    return selected, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exp_ids", nargs="+", metavar="expNNN", help="Experiment ids to archive.")
    parser.add_argument("--apply", action="store_true", help="Actually move; default is a dry run.")
    args = parser.parse_args()

    experiments, problems = discover()
    if problems:
        print("Fix the frontmatter first (tools/experiments_index.py --check):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    selected, problems = resolve(args.exp_ids, experiments)
    if problems:
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    if not selected:
        print("Nothing to do.")
        return 0

    moves = [build_move(e) for e in selected]
    moving = {m.name for m in moves}

    # A reference from an experiment that is *not* moving keeps the archive alive.
    staying = [e for e in experiments if not e.archived and e.name not in moving]
    blocking = []
    for exp in staying:
        config = EXPERIMENTS_DIR / exp.rel_dir / "config.yaml"
        if not config.exists():
            continue  # a tool-driven experiment with no submitted job, e.g. exp087
        text = config.read_text()
        for move in moves:
            if move.old_rel in text:
                blocking.append(f"{exp.name}/config.yaml references {move.name}")
    if blocking:
        print("Refusing — a live config still reads data from these:", file=sys.stderr)
        for line in blocking:
            print(f"  {line}", file=sys.stderr)
        return 1

    hits = find_references(moves, reference_files())
    report_plan(moves, hits, args.apply)

    if not args.apply:
        print("\nRe-run with --apply to perform the move.")
        return 0

    print_migration_footer(apply_moves(moves, hits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
