#!/usr/bin/env python3
"""Split the face-identity prompt masters into the per-identity control sets the trainer needs,
and enforce the anti-cheat rule that the published eval prompts never leak into training.

Mirrors ``split_imagenet_prompts.py``. ``zml/unlearn/unlearn_frame_replace.py`` (via
``control_concept_prompts``) takes one CSV per control set, so a face erase run needs

    prompts/face_identities/<slug>.csv         the 30 published eval prompts of the erased identity
    prompts/face_identities/others_<slug>.csv  one preservation prompt per *other* identity

Both are written once and committed, so no training run re-derives them and every run scores the
same (prompt, seed) pairs (repo seed policy). "others" rows come from ``face_preservation.csv``
(the training retention anchors), not from the eval set, so the live-eval collateral check stays
disjoint from what the model is trained to preserve toward — same reasoning as the imagenet side.

**Hard anti-cheat rule** (``docs/face_identity.md`` §4.6): the 150 published eval prompts
(``prompts/face_cogvideox.csv``, fetched verbatim from T2VUnlearning — see
``tools/fetch_face_eval_prompts.py``) must never appear in a ``prompts/face_identities/split/*.csv`` dataset-
construction file. This script asserts that disjointness (normalized-text comparison) every time it
runs, so a future split_face CSV that accidentally copies an eval prompt fails loudly here instead
of silently inflating a reported number.

Run:
    uv run python tools/split_face_prompts.py
"""

import argparse
import glob
import re
from pathlib import Path

import pandas as pd

from zml.benchmarks.face_identities import FACE_IDENTITIES, identity_slug

EVAL_CSV = "prompts/face_cogvideox.csv"
PRESERVATION_CSV = "prompts/face_preservation.csv"
SPLIT_GLOB = "prompts/face_identities/split/*.csv"  # dataset-construction A/B/C CSVs, must never touch EVAL_CSV
OUTPUT_DIR = "prompts/face_identities"


def _normalize(text: str) -> str:
    """Collapse whitespace and case so a trivial reformatting can't hide an eval-prompt leak."""
    return re.sub(r"\s+", " ", text.strip().lower())


def assert_no_eval_leakage(eval_csv: str, split_glob: str) -> None:
    eval_df = pd.read_csv(eval_csv)
    eval_prompts = {_normalize(p) for p in eval_df["prompt"]}

    for split_path in sorted(glob.glob(split_glob)):
        split_df = pd.read_csv(split_path)
        if "prompt_a" not in split_df.columns:
            raise ValueError(f"{split_path} has no 'prompt_a' column; is this really an A/B/C CSV?")
        leaked = {p for p in split_df["prompt_a"] if _normalize(p) in eval_prompts}
        if leaked:
            raise ValueError(
                f"{split_path} contains {len(leaked)} prompt(s) that also appear in {eval_csv} "
                f"(the published eval set). Training or dataset construction must never use eval "
                f"prompts — see docs/face_identity.md §4.6. Offending prompt(s): {sorted(leaked)[:3]}"
            )
    print(f"Anti-cheat check passed: no {split_glob} prompt_a overlaps {eval_csv}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_csv", default=EVAL_CSV)
    parser.add_argument("--preservation_csv", default=PRESERVATION_CSV)
    parser.add_argument("--split_glob", default=SPLIT_GLOB)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    assert_no_eval_leakage(args.eval_csv, args.split_glob)

    eval_df = pd.read_csv(args.eval_csv)
    preservation_df = pd.read_csv(args.preservation_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One representative preservation prompt per identity; taking the first keeps the choice
    # deterministic and the file diffable. Generic (unnamed-person) anchors are excluded here —
    # "others" means the other *identities* specifically, mirroring imagenet's per-class picks.
    identity_preservation = preservation_df[preservation_df["class_name"] != "generic"]
    representatives = identity_preservation.groupby("class_name", sort=False).head(1)

    for identity_name in FACE_IDENTITIES:
        slug = identity_slug(identity_name)
        concept = eval_df[eval_df["class_name"] == identity_name]
        others = representatives[representatives["class_name"] != identity_name]
        if concept.empty:
            raise ValueError(f"{args.eval_csv} has no rows for {identity_name!r}.")
        if len(others) != len(FACE_IDENTITIES) - 1:
            raise ValueError(
                f"Expected {len(FACE_IDENTITIES) - 1} 'others' anchors for {identity_name!r}, got "
                f"{len(others)} — check {args.preservation_csv} covers every identity."
            )

        concept.to_csv(out_dir / f"{slug}.csv", index=False)
        others.to_csv(out_dir / f"others_{slug}.csv", index=False)
        print(f"{identity_name}: {len(concept)} eval prompts, {len(others)} others")


if __name__ == "__main__":
    main()
