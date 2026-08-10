"""Build the two VBench prompt sets T2VUnlearning reports, in our eval-CSV format.

T2VUnlearning (arXiv:2505.17550) backs its "preserves generation capability" claim with exactly two
VBench dimensions — **Object Class** and **Subject Consistency** — so those are the utility columns
a reviewer will expect from us. Reporting only our own instruments (DOVER, motion, CLIP,
colorfulness), on which our method looks worse, invites the objection that theirs were left out on
purpose. See ``docs/comparability_t2vunlearning.md``.

Both are *general capability* probes on their own prompt sets, not something scorable on the nudity
clips we already have — so each needs its own generation run:

- ``object_class`` (79 prompts) — bare COCO nouns: "a person", "a bicycle", "a car". VBench scores
  them by detecting whether the named object actually appears.
- ``subject_consistency`` (72 prompts) — "a person doing X". VBench scores DINO feature similarity
  across frames; implemented faithfully in ``zml/eval/subject_consistency.py``.

Source is ``VBench_full_info.json`` as **redistributed in T2VUnlearning's own repo**
(``evaluation/vbench_prompts/``), so we take the prompts through the same copy they used rather than
from upstream VBench, and prompt drift between VBench releases cannot silently break the comparison.

VBench itself samples 5 videos per prompt; we generate 1, which is the cost-driven deviation to
state in the paper. Seeds are hash-derived (VBench ships none) and frozen by committing the CSVs,
per the seed policy in CLAUDE.md.

Run:
    uv run python tools/build_vbench_prompts.py
"""

import argparse
import io
import json
import urllib.request

import pandas as pd

from tools.build_external_nudity_evalsets import _stable_seed

VBENCH_INFO_URL = (
    "https://raw.githubusercontent.com/VDIGPKU/T2VUnlearning/main/"
    "evaluation/vbench_prompts/VBench_full_info.json"
)

# The two dimensions T2VUnlearning report. VBench defines sixteen; the rest are out of scope.
DIMENSIONS = {
    "object_class": "prompts/vbench_object_class.csv",
    "subject_consistency": "prompts/vbench_subject_consistency.csv",
}


def _fetch_info() -> list[dict]:
    with urllib.request.urlopen(VBENCH_INFO_URL, timeout=300) as response:
        return json.load(io.BytesIO(response.read()))


def build_dimension(entries: list[dict], dimension: str, output_path: str) -> None:
    # Preserve VBench's own row order rather than sorting, so the CSV mirrors the source file.
    prompts = [e["prompt_en"].strip() for e in entries if dimension in e["dimension"]]
    if not prompts:
        raise ValueError(f"No prompts found for VBench dimension {dimension!r}.")

    pd.DataFrame({
        "prompt": prompts,
        "seed": [_stable_seed(p) for p in prompts],
        "concept": "utility",
        "concept_type": "vbench",
        "source": f"vbench-{dimension.replace('_', '-')}",
        "vbench_dimension": dimension,
    }).to_csv(output_path, index=False)
    print(f"VBench {dimension}: {len(prompts)} prompts -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.parse_args()
    info = _fetch_info()
    for dimension, path in DIMENSIONS.items():
        build_dimension(info, dimension, path)
