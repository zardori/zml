#!/usr/bin/env python3
"""Write the subset of a split-prompt A/B/C CSV named by a list of seeds.

Sampler sweeps run on a handful of rows chosen for a measured property — the five highest-confidence
rows (``chain_saw_closeup_sweep.csv``), or the rows a screen says failed a particular way. Selecting
those by hand-copying prompt text is how a sweep silently stops being a subset of the dataset it is
supposed to inform; selecting them here keeps the rows byte-identical to the source and records the
seed list in the file that produced it.

Seeds are emitted in the order given, and a seed missing from the source is an error rather than a
silently shorter file.

    uv run python tools/subset_split_prompts.py \\
        --source prompts/imagenet_objects/split/chain_saw_closeup.csv \\
        --out prompts/imagenet_objects/split/chain_saw_closeup_suppressed.csv \\
        --seeds 3203 3205 3206
"""

import argparse
import csv
from pathlib import Path

FIELDNAMES = ["prompt_a", "prompt_b", "prompt_c", "seed"]


def subset(source: Path, seeds: list[int]) -> list[dict[str, str]]:
    with source.open(newline="") as handle:
        by_seed = {int(row["seed"]): row for row in csv.DictReader(handle)}
    missing = [s for s in seeds if s not in by_seed]
    if missing:
        raise SystemExit(f"{source} has no row for seed(s) {missing}.")
    return [by_seed[s] for s in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="split-prompt CSV to take rows from")
    parser.add_argument("--out", required=True, type=Path, help="where to write the subset")
    parser.add_argument("--seeds", required=True, type=int, nargs="+", help="seeds to keep, in output order")
    args = parser.parse_args()

    rows = subset(args.source, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{args.out}: {len(rows)} triples (seeds {args.seeds})")


if __name__ == "__main__":
    main()
