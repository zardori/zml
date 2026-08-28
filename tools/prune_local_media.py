#!/usr/bin/env python3
"""Reclaim local disk by pruning generated media that the clusters still hold.

Videos are gitignored build output, not source: every ``.mp4`` under ``experiments/`` was
produced by a job and still lives in that job's cluster repo root, so a local copy is a
cache. ``pull_results.sh`` refills anything this removes.

Three buckets, each independently selectable with ``--bucket``:

``archive``
    Videos under ``experiments/archive/`` — retired threads. Their conclusions live in
    ``docs/`` and their ``metrics.jsonl``/``summary.json`` stay untouched.
``superseded``
    Videos in live-tree experiments whose ``notes.md`` frontmatter says ``superseded`` or
    ``abandoned`` — the registry already calls these retirable.
``dedupe``
    Byte-identical videos that exist at several paths (eval clips copied between
    experiments). These are *hardlinked*, not deleted: nothing is lost and every path keeps
    resolving, so this bucket is safe even for live experiments.

Dry run by default; ``--apply`` performs the work. Idempotent — re-run it after each
``pull_results.sh``.

Registry fields and the archive policy: ``docs/experiment_registry.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from experiments_index import (
    ARCHIVE_DIR,
    EXPERIMENTS_DIR,
    EXP_DIR_RE,
    REPO_ROOT,
    RETIRED_STATUSES,
    parse_frontmatter,
)

MEDIA_SUFFIXES = (".mp4",)
HASH_CHUNK_BYTES = 1 << 20  # 1 MiB; files are ~300 KB, so this is one read each
BUCKETS = ("archive", "superseded", "dedupe")
MB = 1024 * 1024
GB = 1024 * MB


def human(num_bytes: int) -> str:
    if num_bytes >= GB:
        return f"{num_bytes / GB:.2f} GB"
    return f"{num_bytes / MB:.1f} MB"


@dataclass
class Plan:
    """What one bucket proposes to do, accumulated before anything touches the disk."""

    bucket: str
    delete: list[Path] = field(default_factory=list)
    # (redundant copy, file it will point at) — same content, kept as a hardlink.
    link: list[tuple[Path, Path]] = field(default_factory=list)

    @property
    def freed_bytes(self) -> int:
        paths = self.delete + [dup for dup, _ in self.link]
        return sum(p.stat().st_size for p in paths if p.exists())

    @property
    def count(self) -> int:
        return len(self.delete) + len(self.link)


def media_files(root: Path) -> list[Path]:
    """Every generated video under ``root``, sorted for deterministic output."""
    if not root.is_dir():
        return []
    found = (p for suffix in MEDIA_SUFFIXES for p in root.rglob(f"*{suffix}"))
    return sorted(p for p in found if p.is_file() and not p.is_symlink())


def live_experiment_dirs() -> list[Path]:
    """Experiment folders in the live tree (``experiments/<thread>/expNNN_*``)."""
    dirs = [
        exp
        for thread in EXPERIMENTS_DIR.iterdir()
        if thread.is_dir() and thread != ARCHIVE_DIR
        for exp in thread.iterdir()
        if exp.is_dir() and EXP_DIR_RE.match(exp.name)
    ]
    return sorted(dirs)


def plan_archive() -> Plan:
    return Plan("archive", delete=media_files(ARCHIVE_DIR))


def plan_superseded() -> tuple[Plan, list[str]]:
    """Videos of live-tree experiments the registry marks superseded or abandoned."""
    plan, problems = Plan("superseded"), []
    for exp_dir in live_experiment_dirs():
        data, errors = parse_frontmatter(exp_dir / "notes.md")
        if errors:
            problems.extend(errors)
            continue
        if data.get("status") in RETIRED_STATUSES:
            plan.delete.extend(media_files(exp_dir))
    return plan, problems


def file_digest(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_identical(paths: list[Path]) -> list[list[Path]]:
    """Group paths by content, hashing only within same-size candidates."""
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        by_size[path.stat().st_size].append(path)

    groups: list[list[Path]] = []
    for candidates in by_size.values():
        if len(candidates) < 2:
            continue
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for path in candidates:
            by_hash[file_digest(path)].append(path)
        groups.extend(group for group in by_hash.values() if len(group) > 1)
    return groups


def already_linked(a: Path, b: Path) -> bool:
    """True when both paths are the same inode — a previous run already linked them."""
    sa, sb = a.stat(), b.stat()
    return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)


def plan_dedupe(skip: set[Path]) -> Plan:
    """Hardlink redundant copies onto one keeper, ignoring files another bucket removes."""
    plan = Plan("dedupe")
    candidates = [p for p in media_files(EXPERIMENTS_DIR) if p not in skip]
    for group in group_identical(candidates):
        keeper, *duplicates = group  # sorted, so the keeper is stable across runs
        plan.link.extend(
            (dup, keeper) for dup in duplicates if not already_linked(dup, keeper)
        )
    return plan


def relink(duplicate: Path, keeper: Path) -> None:
    """Replace ``duplicate`` with a hardlink to ``keeper`` without a window of absence."""
    staging = duplicate.with_name(duplicate.name + ".prune-tmp")
    os.link(keeper, staging)
    os.replace(staging, duplicate)


def execute(plan: Plan) -> None:
    for path in plan.delete:
        path.unlink(missing_ok=True)
    for duplicate, keeper in plan.link:
        relink(duplicate, keeper)


def summarise(plan: Plan, *, verbose: bool) -> None:
    verb = "hardlink" if plan.bucket == "dedupe" else "delete"
    print(f"  {plan.bucket:<12} {plan.count:>6} files  {human(plan.freed_bytes):>10}  ({verb})")
    if not verbose:
        return
    for path in plan.delete:
        print(f"      - {path.relative_to(REPO_ROOT)}")
    for duplicate, keeper in plan.link:
        print(f"      = {duplicate.relative_to(REPO_ROOT)} -> {keeper.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bucket",
        action="append",
        choices=BUCKETS,
        help="bucket to prune; repeatable (default: all of %s)" % ", ".join(BUCKETS),
    )
    parser.add_argument("--apply", action="store_true", help="perform the work (default: dry run)")
    parser.add_argument("-v", "--verbose", action="store_true", help="list every affected file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    buckets = tuple(dict.fromkeys(args.bucket)) if args.bucket else BUCKETS

    plans: list[Plan] = []
    problems: list[str] = []
    if "archive" in buckets:
        plans.append(plan_archive())
    if "superseded" in buckets:
        plan, issues = plan_superseded()
        plans.append(plan)
        problems.extend(issues)
    if "dedupe" in buckets:
        # Files an earlier bucket removes must not anchor or inflate the dedupe plan.
        doomed = {path for plan in plans for path in plan.delete}
        plans.append(plan_dedupe(doomed))

    print(f"{'apply' if args.apply else 'dry run'}: {', '.join(buckets)}\n")
    for plan in plans:
        summarise(plan, verbose=args.verbose)

    total_files = sum(plan.count for plan in plans)
    total_bytes = sum(plan.freed_bytes for plan in plans)
    print(f"\n  {'total':<12} {total_files:>6} files  {human(total_bytes):>10}")

    for problem in problems:
        print(f"  warning: {problem}", file=sys.stderr)

    if not args.apply:
        print("\nNothing changed. Re-run with --apply to reclaim it.")
        print("Videos are cluster-backed build output; pull_results.sh refills them.")
        return 0

    for plan in plans:
        execute(plan)
    print(f"\nReclaimed {human(total_bytes)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
