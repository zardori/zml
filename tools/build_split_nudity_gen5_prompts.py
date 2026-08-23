"""Build `prompts/split_nudity_gen5.csv` — gen4's scenes with a deliberately *uniform* wardrobe.

Why a fifth generation
----------------------
gen4 was built to be realistic and, as a by-product, maximally diverse: 40 distinct garments across
8 categories, each used ~5 times. It never beat exp080's 34-clip gen1-gen3 set. `docs/frame_replace.md`
records that the LAB edit statistics do **not** explain that gap — gen4's edits are larger, and its
lower coherence (0.605 vs 0.714) is inside the bootstrap spread of 34-clip subsets. So this build is
not implied by a measured finding; it is a **deliberate manipulation of the one variable that has any
mechanistic story attached to it**, run large enough that the coherence difference cannot be noise.

The hypothesis: a LoRA can only realise the component of ``donor - teacher`` that recurs across
examples. Forty different garments means forty different edits and a shared component that is
whatever survives averaging. One garment means every clip's edit is "put *this* on the subject",
which should raise coherence far outside the n=34 noise band.

**Screen before training.** ``tools/analyze_edit_directions.py`` reads coherence off the built clips
with no GPU, so the manipulation is verified before a training job is spent:

    uv run python tools/analyze_edit_directions.py --metadata <gen5 metadata.json> \
        --videos <gen5 videos dir> --label GEN5

If coherence does not clear gen4's 0.605 + 2 sd (~0.85 at n=34, or simply "well above 0.714 at
n>=75"), the manipulation failed at the data level and no training run is warranted.

What is held identical to gen4, so the comparison is single-variable
-------------------------------------------------------------------
The 25 scenes, the A and C prompt construction, the locked-camera grammar, the banned-wardrobe list
and the 200-row size all come from ``build_split_nudity_gen4_prompts`` by import. Only ``GARMENTS``
changes: 3 near-identical navy pieces instead of 40 varied ones.

Why three and not one
---------------------
"One garment" is the cleanest manipulation but the worst collateral risk: the adapter could learn
"render this exact shirt" rather than "do not render nudity". exp113 already caught the gen4
checkpoint shifting colour on prompts containing no nudity (53.1 vs base 45.8), and a uniform donor
set can only make that stronger. Three pieces in one colour family and one silhouette keep coherence
near-maximal while giving the model *some* within-concept variation to generalise over. **The eval
for any run on this set must include the unrelated/preservation prompts**, or the collateral this
trades for will not be visible.

Seeds are 5601-5800 — verified free of every other prompt CSV (gen4 is 3801-4000, the imagenet sets
run to 4330, the face sets scatter up to 5593). Commit the CSV and never renumber it.

Run:
    uv run python tools/build_split_nudity_gen5_prompts.py
"""

import argparse
import csv
from pathlib import Path

from build_split_nudity_gen4_prompts import (
    BANNED_SUBSTRINGS,
    SCENES,
    Garment,
    Row,
    _write,
)

OUTPUT_DEFAULT = Path("prompts/split_nudity_gen5.csv")
FIRST_SEED = 5601
REPEATS = 8  # 25 scenes x 8 = 200 rows, matching gen4's size exactly

# The colour token every garment must contain. Asserted in verify(): the entire point of this
# generation is that the donor wardrobe does not vary in colour, and a future edit that quietly
# adds a green option would destroy the manipulation without failing anything.
COLOUR_TOKEN = "navy"

# Chosen against the two failure modes earlier generations hit, not from colour theory (the
# chroma story that briefly motivated a colour choice was retracted — see docs/frame_replace.md):
#   * gen1-gen3 shipped cream/beige sacks that read as skin  -> navy cannot be confused with skin.
#   * gen1-gen3 shipped bulk-for-coverage's sake             -> these are ordinary fitted clothes.
# Full coverage still comes from naming garments that cover (long sleeves, full-length legwear),
# never from a "no bare skin" constraint.
GARMENTS: tuple[Garment, ...] = (
    Garment("navy_shirt", "a plain navy long-sleeved cotton shirt and straight-leg dark indigo jeans",
            "a plain navy long-sleeved cotton shirt"),
    Garment("navy_sweater", "a plain navy crewneck sweater and straight-leg dark indigo jeans",
            "a plain navy crewneck sweater"),
    Garment("navy_buttonup", "a plain navy button-up shirt with the sleeves to the wrist and dark indigo trousers",
            "a plain navy button-up shirt with the sleeves to the wrist"),
)


def build_rows() -> list[Row]:
    """25 scenes x REPEATS, the garment rotating so every scene is seen in all three."""
    rows: list[Row] = []
    for repeat in range(REPEATS):
        for scene_index, scene in enumerate(SCENES):
            garment = GARMENTS[(repeat + scene_index) % len(GARMENTS)]
            rows.append(
                Row(
                    prompt_a=scene.prompt_a(),
                    prompt_b=scene.prompt_b(garment),
                    prompt_c=scene.prompt_c(),
                    seed=FIRST_SEED + len(rows),
                    category=garment.category,
                )
            )
    return rows


def verify(rows: list[Row]) -> None:
    """Fail the build rather than ship a set that is not actually uniform."""
    if len(GARMENTS) > 3:
        raise SystemExit(f"gen5 is the uniform-wardrobe arm; {len(GARMENTS)} garments is not uniform")
    for garment in GARMENTS:
        for text in (garment.full, garment.upper):
            if COLOUR_TOKEN not in text.lower():
                raise SystemExit(f"Garment is off the gen5 colour family ('{COLOUR_TOKEN}'): {text}")

    for row in rows:
        lowered = row.prompt_b.lower()
        for banned in BANNED_SUBSTRINGS:
            if banned in lowered:
                raise SystemExit(f"Banned wardrobe '{banned}' in seed {row.seed}: {row.prompt_b}")

    seeds = [r.seed for r in rows]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("Duplicate seeds")

    per_scene = {r.prompt_c: sum(1 for x in rows if x.prompt_c == r.prompt_c) for r in rows}
    if len(set(per_scene.values())) != 1:
        raise SystemExit(f"Scenes are unbalanced: {sorted(set(per_scene.values()))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--parts", type=int, default=4,
                        help="Also write N round-robin shards for grid submission (0 to skip)")
    args = parser.parse_args()

    rows = build_rows()
    verify(rows)
    _write(args.out, rows)

    per_garment = {c: sum(1 for r in rows if r.category == c) for c in sorted({r.category for r in rows})}
    print(f"Wrote {len(rows)} rows -> {args.out}  (seeds {rows[0].seed}-{rows[-1].seed})")
    print(f"  {len(set(r.prompt_b for r in rows))} distinct B prompts "
          f"(gen4 has 200; that difference is the manipulation)")
    for garment, count in per_garment.items():
        print(f"  {garment:<16} {count}")

    for part in range(args.parts):
        shard = [r for i, r in enumerate(rows) if i % args.parts == part]
        path = args.out.with_name(f"{args.out.stem}_part{part + 1}{args.out.suffix}")
        _write(path, shard)
        print(f"  {path}  {len(shard)} rows, {len({r.category for r in shard})}/{len(per_garment)} garments")


if __name__ == "__main__":
    main()
