#!/usr/bin/env python3
"""Split the ImageNet object prompt masters into the per-class control sets the trainer needs.

``zml/unlearn/unlearn_frame_replace.py`` takes one CSV per control set, so erasing a class needs

    prompts/imagenet_objects/<slug>.csv         the 20 eval prompts of the erased class
    prompts/imagenet_objects/others_<slug>.csv  one preservation prompt per *other* class

Both are written once and committed, so no training run re-derives them and every run scores the
same (prompt, seed) pairs (repo seed policy). The "others" rows come from the preservation pool, not
from the 20 eval prompts, so the live-eval collateral check stays disjoint from the eval set.

Run:  uv run python tools/split_imagenet_prompts.py
"""

import argparse
from pathlib import Path

import pandas as pd

from zml.benchmarks.imagenet_classes import IMAGENETTE_CLASSES, class_slug

EVAL_CSV = "prompts/imagenet_objects.csv"
PRESERVATION_CSV = "prompts/imagenet_preservation.csv"
OUTPUT_DIR = "prompts/imagenet_objects"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_csv", default=EVAL_CSV)
    parser.add_argument("--preservation_csv", default=PRESERVATION_CSV)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    eval_df = pd.read_csv(args.eval_csv)
    preservation_df = pd.read_csv(args.preservation_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One representative preservation prompt per class; taking the first keeps the choice
    # deterministic and the file diffable.
    representatives = preservation_df.groupby("class_name", sort=False).head(1)

    for class_name in IMAGENETTE_CLASSES:
        slug = class_slug(class_name)
        concept = eval_df[eval_df["class_name"] == class_name]
        others = representatives[representatives["class_name"] != class_name]
        if concept.empty:
            raise ValueError(f"{args.eval_csv} has no rows for {class_name!r}.")

        concept.to_csv(out_dir / f"{slug}.csv", index=False)
        others.to_csv(out_dir / f"others_{slug}.csv", index=False)
        print(f"{class_name}: {len(concept)} concept prompts, {len(others)} others")


if __name__ == "__main__":
    main()
