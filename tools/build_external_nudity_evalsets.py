"""Build nudity eval prompt CSVs from *external, published* benchmarks.

Why this exists: every nudity number the project has so far is measured on
``prompts/cogvideox_nudity.csv``, which we wrote ourselves. It is described elsewhere as
"i2p-derived", but it shares **zero** prompts with the actual I2P benchmark — it is I2P-*styled*
hand-written text. That is a problem for a paper on two counts: the numbers cannot be placed in a
table next to published work that evaluates on real benchmarks, and having authored both the
training prompts and the eval prompts invites the objection that the eval vocabulary happens to
resemble what we trained on. Scoring on sets nobody on the team wrote removes both objections.

This script downloads the published sources and writes committed CSVs in the project's eval-prompt
format (``prompt,seed`` + provenance columns). It is deterministic: rerunning it reproduces the
same rows in the same order with the same seeds, so the committed CSVs can always be regenerated.

Sources
-------
**I2P** (Inappropriate Image Prompts, Schramowski et al., CVPR 2023) — the standard prompt set used
by essentially every concept-erasure paper. Fetched from the Ring-A-Bell repo's mirror
(``data/unsafe-prompts4703.csv``), which is the same 4703-prompt release. Two properties make it
ideal here:

- ``nudity_percentage`` — the measured fraction of reference SD-1.4 samples for that prompt that
  contained nudity. Filtering on it selects prompts that *reliably* elicit the concept, which is
  what an erasure metric needs, and it is a property of the benchmark rather than our judgement.
- ``evaluation_seed`` — the benchmark ships its own per-prompt seed, so the project seed policy
  (CLAUDE.md: eval seeds live in the CSV, fixed forever) is satisfied by the benchmark's seeds
  rather than any we invent.

Caveat to state in the paper: I2P prompts were written for a text-to-*image* model (SD-1.4), so
they are comma-separated art-style prompts, often naming artists. Published T2V unlearning work
reuses them as-is and so do we, but they are out-of-distribution for a T2V model's usual caption
style — which is exactly why the second set below is worth having.

**SafeSora** (PKU-Alignment, CC-BY-NC-4.0) — a text-to-*video* safety dataset, so its prompts are
in-distribution for T2V. Its released config files label prompts only ``safety_critical`` /
``safety_neutral``, with **no per-harm-category annotation**, so the nudity subset here is selected
by our own documented keyword filter (``SAFESORA_NUDITY_KEYWORDS``) over the safety-critical
prompts. That filter is ours, not SafeSora's taxonomy — say so in the paper rather than implying
SafeSora defines this split. SafeSora ships no seeds, so seeds are assigned here deterministically
(hash of the prompt, see ``_stable_seed``) and then frozen by committing the CSV.

**Ring-A-Bell** is deliberately *not* built here: the repo releases inverted/adversarial prompts
for Violence only (``data/InvPrompt/Violence/``), not for nudity. Generating nudity adversarial
prompts would mean running their genetic-algorithm attack against our text encoder using the
released ``Nudity_vector.npy`` — a real implementation task, not a download. Tracked as possible
future robustness work, not claimed as available.

Run:
    uv run python tools/build_external_nudity_evalsets.py
"""

import argparse
import gzip
import hashlib
import io
import json
import urllib.request

import pandas as pd

I2P_URL = "https://raw.githubusercontent.com/chiayi-hsu/Ring-A-Bell/main/data/unsafe-prompts4703.csv"
SAFESORA_TEST_URL = (
    "https://huggingface.co/datasets/PKU-Alignment/SafeSora/resolve/main/config-test.json.gz"
)

# Keep an I2P prompt when more than this share of the benchmark's reference SD-1.4 samples for it
# contained nudity. 50 selects prompts that elicit the concept more often than not (95 of 4703),
# which is both a defensible "this prompt really is a nudity prompt" line and a set size comparable
# to the existing 100-prompt cogvideox_nudity.csv.
I2P_NUDITY_PERCENTAGE_MIN = 50.0

# Our filter over SafeSora's safety-critical prompts (SafeSora publishes no harm-category labels).
# Deliberately explicit/anatomical terms only — the goal is prompts whose *intent* is sexual nudity,
# not every prompt mentioning a body part.
SAFESORA_NUDITY_KEYWORDS = (
    "nude", "naked", "nudity", "topless", "undress", "strip",
    "sex", "sexual", "porn", "erotic", "orgasm", "masturbat",
    "breast", "boob", "nipple", "genital", "penis", "vagina",
    "threesome", "blowjob", "intercourse", "fetish", "lingerie",
)
# Cap so one eval job stays comparable in cost to the others (~2 min/clip at 49 frames/50 steps).
SAFESORA_MAX_ROWS = 100

SEED_MODULUS = 1_000_000  # keep assigned seeds in the same magnitude as existing prompt CSVs


def _stable_seed(text: str) -> int:
    """Deterministic per-prompt seed for sources that ship none (SafeSora).

    Hash-derived rather than positional so that adding/removing/reordering rows never silently
    changes which seed an existing prompt gets — the exact failure the seed policy in CLAUDE.md
    warns about. Frozen in practice by committing the resulting CSV.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % SEED_MODULUS


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def build_i2p_nudity(output_path: str) -> None:
    df = pd.read_csv(io.BytesIO(_fetch(I2P_URL)))
    kept = df[df["nudity_percentage"] > I2P_NUDITY_PERCENTAGE_MIN].copy()
    # Sort by the benchmark's own case_number so the row order is the benchmark's, not ours.
    kept = kept.sort_values("case_number")
    out = pd.DataFrame({
        "prompt": kept["prompt"].astype(str).str.strip(),
        "seed": kept["evaluation_seed"].astype(int),  # I2P's own seed, not ours
        "concept": "nudity",
        "concept_type": "safety",
        "source": "i2p",
        "i2p_case_number": kept["case_number"].astype(int),
        "i2p_nudity_percentage": kept["nudity_percentage"].astype(float),
        "i2p_categories": kept["categories"].astype(str),
    })
    out.to_csv(output_path, index=False)
    print(f"I2P nudity: {len(out)} prompts (nudity_percentage > {I2P_NUDITY_PERCENTAGE_MIN}) -> {output_path}")


def build_safesora_nudity(output_path: str) -> None:
    records = json.load(gzip.open(io.BytesIO(_fetch(SAFESORA_TEST_URL))))

    # One row per unique prompt: the file is a preference dataset with many video pairs per prompt.
    seen: dict[str, str] = {}
    for record in records:
        if record.get("prompt_type") != "safety_critical":
            continue
        prompt = str(record.get("prompt_text", "")).strip()
        if prompt:
            seen.setdefault(record["prompt_id"], prompt)

    matched = sorted(
        (pid, text) for pid, text in seen.items()
        if any(keyword in text.lower() for keyword in SAFESORA_NUDITY_KEYWORDS)
    )
    # Deterministic subset: order by the seed we assign, so the cap is not "whatever sorted first"
    # (which would bias toward prompts starting with punctuation/digits).
    ranked = sorted(matched, key=lambda item: _stable_seed(item[1]))[:SAFESORA_MAX_ROWS]

    out = pd.DataFrame({
        "prompt": [text for _, text in ranked],
        "seed": [_stable_seed(text) for _, text in ranked],
        "concept": "nudity",
        "concept_type": "safety",
        "source": "safesora",
        "safesora_prompt_id": [pid for pid, _ in ranked],
    })
    out.to_csv(output_path, index=False)
    print(
        f"SafeSora nudity: {len(out)} prompts "
        f"(of {len(matched)} keyword-matched safety_critical) -> {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i2p_out", default="prompts/i2p_nudity.csv")
    parser.add_argument("--safesora_out", default="prompts/safesora_nudity.csv")
    args = parser.parse_args()
    build_i2p_nudity(args.i2p_out)
    build_safesora_nudity(args.safesora_out)
