"""Build the eval prompt CSVs released by T2VUnlearning (arXiv:2505.17550), our closest comparison.

T2VUnlearning is the paper we most need to sit beside: same base model (CogVideoX-5B), same
v-prediction parameterization, and it reports nudity, ImageNet objects and celebrity identity. It
publishes its evaluation prompt sets at ``evaluation/data/`` in the official repo, which means the
"different prompts, therefore incomparable" objection is avoidable by simply using theirs.

What this script establishes
----------------------------
1. **Verification, not a build.** ``prompts/cogvideox_nudity.csv`` — the in-house set every nudity
   number in this project is measured on, and which ``docs/external_eval_sets.md`` described as
   something "we wrote" — is byte-for-byte T2VUnlearning's ``nudity_cogvideox.csv``: the same 100
   prompts, in the same order, with the same 100 seeds. It is *their released Gen set*. That is
   worth a great deal (our whole historical series is already on the paper's own prompts and seeds)
   and it is worth pinning in code, so ``verify_gen_set`` re-checks it and fails loudly on drift.

2. **Ring-A-Bell nudity.** ``docs/external_eval_sets.md`` recorded that these are unavailable —
   true of the Ring-A-Bell repo itself, which releases inverted prompts for Violence only. But
   T2VUnlearning ships the nudity ones it used (79 rows) in its own repo, so the Ring-A-Bell column
   of their Table 1 *is* reproducible by download. Written to ``prompts/ring_a_bell_nudity.csv``.

3. **A free paired preservation set.** That file has a second column, ``normal prompt``: a safe
   rewrite of each adversarial prompt ("a woman in a bedroom, bare and exposed" ->
   "...fully clothed and serene"). Written to ``prompts/ring_a_bell_nudity_safe.csv`` with **the
   same seed as its partner**, so the pair differs only in the prompt. That is exactly the
   held-out ``related`` signal nudity has been missing, and the pairing makes it stronger than an
   unpaired set: a rate that falls on the adversarial prompt while holding on its own safe rewrite
   is erasure, and one that falls on both is collateral damage.

4. **Their SafeSora subset.** 33 prompts, ``prompts/t2vunlearning_safesora_nudity.csv``. Note this
   shares **zero** prompts with our ``prompts/safesora_nudity.csv`` (100 prompts from our own
   keyword filter) — both are legitimate but they are different sets, so a number on one may not be
   put in a column headed by the other.

Seeds: their Gen set ships seeds and we already use them. Ring-A-Bell and SafeSora ship none, so
seeds are hash-derived (``_stable_seed``, shared with ``build_external_nudity_evalsets``) and frozen
by committing the CSVs, per the seed policy in CLAUDE.md.

Run:
    uv run python tools/build_t2vunlearning_evalsets.py
"""

import argparse
import io
import urllib.request

import pandas as pd

from tools.build_external_nudity_evalsets import _stable_seed

REPO_RAW = "https://raw.githubusercontent.com/VDIGPKU/T2VUnlearning/main/evaluation/data"
GEN_URL = f"{REPO_RAW}/nudity_cogvideox.csv"
RING_A_BELL_URL = f"{REPO_RAW}/nudity-ring-a-bell.csv"
SAFESORA_URL = f"{REPO_RAW}/safe-sora.csv"

# Their CSVs are UTF-8 with a BOM on some files; pandas needs to be told or the first column name
# comes back as "﻿prompt" and every lookup misses.
ENCODING = "utf-8-sig"


def _fetch_csv(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=300) as response:
        return pd.read_csv(io.BytesIO(response.read()), encoding=ENCODING)


def verify_gen_set(local_path: str) -> None:
    """Assert our in-house nudity set is still identical to T2VUnlearning's released Gen set.

    This is the load-bearing claim behind reporting our historical runs in their table, so it is
    checked rather than trusted. Prompts must match in order, and every seed must match, because a
    different seed means a different video for the same prompt.
    """
    theirs = _fetch_csv(GEN_URL)
    ours = pd.read_csv(local_path)

    if len(ours) != len(theirs):
        raise ValueError(f"{local_path} has {len(ours)} rows, T2VUnlearning's Gen set has {len(theirs)}.")
    prompt_mismatch = (ours["prompt"].str.strip() != theirs["prompt"].str.strip()).sum()
    seed_mismatch = (ours["seed"].astype(int) != theirs["seed"].astype(int)).sum()
    if prompt_mismatch or seed_mismatch:
        raise ValueError(
            f"{local_path} has diverged from T2VUnlearning's nudity_cogvideox.csv: "
            f"{prompt_mismatch} prompt(s) and {seed_mismatch} seed(s) differ. Every historical "
            f"nudity number is measured on this file; if it changed, none of them are comparable "
            f"to their Table 1 any more."
        )
    print(f"VERIFIED {local_path}: identical to T2VUnlearning's Gen set ({len(ours)} prompts, seeds match)")


def build_ring_a_bell(unsafe_path: str, safe_path: str) -> None:
    df = _fetch_csv(RING_A_BELL_URL)
    prompts = df["prompt"].astype(str).str.strip()
    safe_prompts = df["normal prompt"].astype(str).str.strip()
    # One seed per *pair*, derived from the adversarial prompt, so the safe rewrite is generated
    # with the same noise as the prompt it is paired with and the two are directly subtractable.
    seeds = [_stable_seed(text) for text in prompts]

    pd.DataFrame({
        "prompt": prompts,
        "seed": seeds,
        "concept": "nudity",
        "concept_type": "safety",
        "source": "ring-a-bell",
        "pair_index": range(len(df)),
    }).to_csv(unsafe_path, index=False)

    pd.DataFrame({
        "prompt": safe_prompts,
        "seed": seeds,
        "concept": "nudity",
        "concept_type": "related",
        "source": "ring-a-bell-normal",
        "pair_index": range(len(df)),
    }).to_csv(safe_path, index=False)

    print(f"Ring-A-Bell nudity: {len(df)} prompts -> {unsafe_path}")
    print(f"Ring-A-Bell paired safe rewrites: {len(df)} prompts -> {safe_path}")


def build_their_safesora(output_path: str) -> None:
    df = _fetch_csv(SAFESORA_URL)
    prompts = df["prompt"].astype(str).str.strip()
    pd.DataFrame({
        "prompt": prompts,
        "seed": [_stable_seed(text) for text in prompts],
        "concept": "nudity",
        "concept_type": "safety",
        "source": "t2vunlearning-safesora",
    }).to_csv(output_path, index=False)
    print(f"T2VUnlearning SafeSora subset: {len(df)} prompts -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gen_set", default="prompts/cogvideox_nudity.csv",
                        help="Local set to verify against their released Gen set (not rewritten)")
    parser.add_argument("--ring_a_bell_out", default="prompts/ring_a_bell_nudity.csv")
    parser.add_argument("--ring_a_bell_safe_out", default="prompts/ring_a_bell_nudity_safe.csv")
    parser.add_argument("--safesora_out", default="prompts/t2vunlearning_safesora_nudity.csv")
    args = parser.parse_args()

    verify_gen_set(args.gen_set)
    build_ring_a_bell(args.ring_a_bell_out, args.ring_a_bell_safe_out)
    build_their_safesora(args.safesora_out)
