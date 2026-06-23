#!/usr/bin/env python3
"""Merge partial-fire prompt CSV files that may have different column sets.

Each input must contain at least the ``prompt`` and ``seed`` columns. The merged
output keeps only the columns common to every input (their intersection), which
is always guaranteed to include ``prompt`` and ``seed``. Rows are deduplicated on
the ``(prompt, seed)`` pair, keeping the first occurrence across files in the
order the files are given.

Example:
    ./tools/merge_partial_fire_prompts.py \\
        prompts/cogvideox_partial_fire.csv \\
        experiments/exp040_partial_fire_search/outputs_*/accepted_pairs.csv \\
        -o prompts/cogvideox_partial_fire.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REQUIRED_COLUMNS: tuple[str, ...] = ("prompt", "seed")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV, returning its field order and rows as dicts."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: file has no header")
        fieldnames = list(reader.fieldnames)
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(f"{path}: missing required column(s): {', '.join(missing)}")
        return fieldnames, list(reader)


def common_columns(field_sets: list[list[str]]) -> list[str]:
    """Columns present in every input, ordered as in the first input."""
    shared = set.intersection(*(set(fs) for fs in field_sets))
    return [c for c in field_sets[0] if c in shared]


def merge(paths: list[Path]) -> tuple[list[str], list[dict[str, str]]]:
    parsed = [read_csv(p) for p in paths]
    field_sets = [fields for fields, _ in parsed]

    columns = common_columns(field_sets)
    # Intersection always includes REQUIRED_COLUMNS since read_csv validated each
    # input contains them, so no extra check is needed here.

    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for (_, rows), path in zip(parsed, paths):
        for row in rows:
            key = (row["prompt"], row["seed"])
            if key in seen:
                continue
            seen.add(key)
            merged.append({c: row[c] for c in columns})

    return columns, merged


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="input CSV files to merge (2 or more)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.inputs) < 2:
        print("error: provide at least two input files to merge", file=sys.stderr)
        return 2

    try:
        columns, rows = merge(args.inputs)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_csv(args.output, columns, rows)
    print(f"merged {len(args.inputs)} files -> {args.output}: {len(rows)} unique rows, columns: {', '.join(columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
