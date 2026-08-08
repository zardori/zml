"""Fetch T2VUnlearning's published face-erasure eval prompts and remap them to repo conventions.

Why this exists: docs/comparison_targets.md's face-identity axis needs an eval set nobody on the
team wrote, for the same reason prompts/i2p_nudity.csv exists instead of using our own
cogvideox_nudity.csv (see docs/external_eval_sets.md). Unlike nudity, T2VUnlearning *does* publish
its face eval prompts directly:

    https://github.com/VDIGPKU/T2VUnlearning/blob/main/evaluation/data/face_cogvideox.csv

150 rows: 30 hand-written scene prompts for each of 5 celebrities (Angela Merkel, Barack Obama,
Donald Trump, Joe Biden, Queen Elizabeth II), with their own `evaluation_seed` per row.

This script downloads that CSV verbatim and remaps its columns to the project's eval-prompt
convention (``prompt,seed`` + provenance), mirroring `imagenet_eval.load_class_prompts`'s expected
`class_name` column so `zml/eval/face_eval.py` can group by identity the same way
`imagenet_eval.py` groups by ImageNet class. It is deterministic: rerunning it reproduces the same
committed CSV, so provenance never has to be reconstructed by hand.

**These prompts are eval-only.** They must never appear in `prompts/split_face_*.csv` (the
frame_replace training/dataset-construction prompts) or in dataset generation of any kind — scoring
on your own training vocabulary is exactly the credibility problem this fetch is meant to avoid.
`tools/split_face_prompts.py` asserts this disjointness at build time.

Run:
    uv run python tools/fetch_face_eval_prompts.py
"""

import argparse
import io
import urllib.request

import pandas as pd

FACE_PROMPTS_URL = (
    "https://raw.githubusercontent.com/VDIGPKU/T2VUnlearning/main/evaluation/data/face_cogvideox.csv"
)

EXPECTED_ROWS = 150
EXPECTED_IDENTITIES = {
    "Angela Merkel": 30,
    "Barack Obama": 30,
    "Donald Trump": 30,
    "Joe Biden": 30,
    "Queen Elizabeth II": 30,
}


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def build_face_eval_prompts(output_path: str, source_url: str = FACE_PROMPTS_URL) -> None:
    raw = pd.read_csv(io.BytesIO(_fetch(source_url)))

    missing = {"case_number", "prompt", "class", "evaluation_seed"} - set(raw.columns)
    if missing:
        raise ValueError(f"{source_url} is missing expected columns: {sorted(missing)}")

    counts = raw["class"].value_counts().to_dict()
    if len(raw) != EXPECTED_ROWS or counts != EXPECTED_IDENTITIES:
        raise ValueError(
            f"Unexpected shape fetched from {source_url}: {len(raw)} rows, per-identity counts "
            f"{counts}; expected {EXPECTED_ROWS} rows with {EXPECTED_IDENTITIES}. The upstream "
            "file may have changed — do not silently adopt a different eval set."
        )

    out = pd.DataFrame({
        "prompt": raw["prompt"].astype(str).str.strip(),
        "seed": raw["evaluation_seed"].astype(int),  # their seed, not ours
        "class_name": raw["class"].astype(str),
        "case_number": raw["case_number"].astype(int),
        "source": "t2vunlearning_face_cogvideox",
    })
    out.to_csv(output_path, index=False)
    print(f"Face eval prompts: {len(out)} rows across {out['class_name'].nunique()} identities -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="prompts/face_cogvideox.csv")
    parser.add_argument("--source_url", default=FACE_PROMPTS_URL)
    args = parser.parse_args()
    build_face_eval_prompts(args.output, args.source_url)
