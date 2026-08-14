#!/usr/bin/env python3
"""File live experiments under ``experiments/<thread>/``, matching the archive's layout.

Why
---
Three concepts are being unlearned in parallel, so experiment *numbers* interleave: the ImageNet
object thread is exp064-exp072, then jumps to exp099 and exp117-exp119, with 30-odd nudity and face
experiments in between. A flat ``experiments/`` makes a thread impossible to see, and the number of
a new experiment says nothing about what it belongs to.

The archive already solved this -- retired work sits in ``experiments/archive/<thread>/``. This
applies the same grouping to live work, so a thread is one directory at every stage of its life and
archiving becomes a move sideways rather than a re-shuffle.

The alternative considered was suffixed numbering (``exp066_1``, ``exp066_2``) to tie a rebuild to
what it rebuilds. It was rejected: it breaks the one-number-one-experiment assumption that
``exp_id`` and every cross-reference in ``docs/`` rely on, it says nothing about the *next*
experiment in a thread (exp120 for faces would still land beside object folders), and "which
experiment supersedes which" is already recorded, in the ``status``/``takeaway`` frontmatter that
``INDEX.md`` renders.

    tools/regroup_experiments.py                    # dry run over every misfiled live experiment
    tools/regroup_experiments.py --apply
    tools/regroup_experiments.py expNNN --apply     # just these

Idempotent: an experiment already in its thread directory is not a move.
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
from experiments_index import discover, Experiment


def build_move(exp: Experiment) -> Move:
    return Move(
        name=exp.name,
        thread=exp.thread,
        old_rel=f"experiments/{exp.rel_dir}",
        new_rel=f"experiments/{exp.thread}/{exp.name}",
    )


def needs_move(exp: Experiment) -> bool:
    """True when a live experiment is not already directly under its own thread directory."""
    return not exp.archived and exp.rel_dir.parts[:-1] != (exp.thread,)


def select(exp_ids: list[str], experiments: list[Experiment]) -> tuple[list[Experiment], list[str]]:
    candidates = [e for e in experiments if not e.archived]
    problems: list[str] = []

    if exp_ids:
        by_id = {e.exp_id: e for e in candidates}
        chosen = []
        for exp_id in exp_ids:
            exp = by_id.get(exp_id)
            if exp is None:
                problems.append(f"{exp_id}: no such live experiment")
            else:
                chosen.append(exp)
    else:
        chosen = candidates

    selected = []
    for exp in chosen:
        if not exp.thread:
            problems.append(f"{exp.exp_id}: no 'thread' in its notes.md frontmatter")
        elif needs_move(exp):
            selected.append(exp)
    return selected, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exp_ids", nargs="*", metavar="expNNN",
                        help="Experiment ids to file; default is every misfiled live experiment.")
    parser.add_argument("--apply", action="store_true", help="Actually move; default is a dry run.")
    args = parser.parse_args()

    experiments, problems = discover()
    if problems:
        print("Fix the frontmatter first (tools/experiments_index.py --check):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    selected, problems = select(args.exp_ids, experiments)
    if problems:
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    if not selected:
        print("Every live experiment is already under its thread directory.")
        return 0

    moves = [build_move(e) for e in selected]
    hits = find_references(moves, reference_files())
    report_plan(moves, hits, args.apply)

    if not args.apply:
        print("\nRe-run with --apply to perform the move.")
        return 0

    print_migration_footer(apply_moves(moves, hits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
