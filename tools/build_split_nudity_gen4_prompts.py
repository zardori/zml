"""Build `prompts/split_nudity_gen4.csv` — 200 A/B/C triples with realistic wardrobe.

Why a fourth generation
-----------------------
Human review of the gen1-gen3 training targets (2026-08-11) found the clothing in the *edited* clips
looks wrong in two specific ways: either implausibly baggy, or close enough to skin tone to read as
bare. Both trace to how the B prompts were written, not to bad luck in generation.

Every gen1-gen3 B prompt was built to satisfy the phrase **"no bare skin visible"**, and the cheapest
way for a model to satisfy that constraint is a shapeless sack. So the set asks for a "long
floor-length robe", a "heavy wool overcoat and thick trousers", a "zipped-up parka", a "thick
turtleneck sweater". We optimised coverage and got un-realism as the by-product — the same shape of
mistake as exp079's retention set, where a proxy constraint quietly determined the composition.
gen3 additionally used "wrapped in white towels" and "matching black leotards", which are exactly the
skin-adjacent wardrobe exp104 banned from the retention set, and the likeliest source of the
"resembles skin" cases.

What changes here:

1. **Garments are named positively and specifically** — "a fitted navy cotton t-shirt and blue
   jeans", not "fully clothed ... no bare skin visible". The negative constraint is gone entirely;
   coverage comes from naming a garment that covers, which is what a person would actually wear.
2. **Nothing bulky.** No robes, parkas, overcoats, heavy winter coats. Outerwear is fitted.
3. **Colour is explicit and varied**, so the target distribution is not uniformly low-chroma. (Note
   the measured colorfulness gap between edited and original clips was only -8.8%, so wardrobe is
   *not* what desaturates our trained model — eta=2 extrapolation is. This is about realism, not
   about the colour metric.)
4. **Banned outright**: towels, robes, leotards, singlets, sports bras, swimwear, sleepwear, bare
   midriffs. `verify()` enforces this and is run at build time.

Scene grammar is held to what already generates well: static locked cameras, the close-up and
multi-person framings exp078 introduced, one to three subjects, no camera motion. 25 scenes x 8
wardrobe categories = 200 rows, each scene appearing once per category so a reviewer can compare
wardrobes against a fixed background rather than against noise.

Seeds are 3801-4000; split_nudity gen1-gen3 occupy 3103-3755, so there is no collision. One seed per
row, shared by A, B, C and the combined clip, per the seed policy in CLAUDE.md — commit the CSV and
never renumber it.

Also writes `--parts` shards alongside the full file. `frame_replace_split_precompute.py` takes a
single `csv_path` and has no offset/limit, but `submit_job.py` grid-searches any list-valued config
field — so a config listing the shards runs them as parallel jobs. exp078 spent ~6.25h on 50
triples, which would be ~25h for 200 in one job; four shards keep each job inside a sane wall clock.
Shards are assigned round-robin so every shard carries all eight wardrobe categories, and a shard
that fails or gets rejected in review does not remove a whole category from the set.

Run:
    uv run python tools/build_split_nudity_gen4_prompts.py
    uv run python tools/build_split_nudity_gen4_prompts.py --out prompts/other.csv --parts 4
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

OUTPUT_DEFAULT = Path("prompts/split_nudity_gen4.csv")
FIRST_SEED = 3801
GARMENTS_PER_CATEGORY = 5

# Wardrobe that is either bulk-for-coverage's sake or close enough to skin to defeat the point.
# gen1-gen3 contain most of these; see the module docstring.
BANNED_SUBSTRINGS = (
    "robe", "parka", "overcoat", "winter coat", "towel", "leotard", "singlet",
    "sports bra", "swimsuit", "bikini", "swim", "pyjama", "pajama", "nightgown",
    "lingerie", "midriff", "bare skin", "no bare skin",
)

# The tail of every prompt. Locked camera is load-bearing: split-prompt conditions two temporal
# regions on different prompts, and camera motion makes the seam impossible to heal.
STILL = "photorealistic. The camera is fixed and never moves."


@dataclass(frozen=True)
class Garment:
    """A wardrobe option. `upper` is used for close-ups that only frame the torso."""

    category: str
    full: str
    upper: str


@dataclass(frozen=True)
class Scene:
    """A shot. Close-ups name a body part; full shots name subjects, pose and setting."""

    shot: str
    closeup: bool
    part: str = ""
    subject: str = ""
    pose: str = ""
    setting: str = ""

    def prompt_a(self) -> str:
        if self.closeup:
            return (
                f"{self.shot} of a completely bare {self.part}, cropped tightly, "
                f"full anatomical detail, {STILL}"
            )
        return (
            f"{self.shot} of {self.subject} completely naked {self.pose} in {self.setting}, "
            f"full anatomical detail, {STILL}"
        )

    def prompt_b(self, garment: Garment) -> str:
        if self.closeup:
            return (
                f"{self.shot} of a {self.part} wearing {garment.upper}, cropped tightly, {STILL}"
            )
        return (
            f"{self.shot} of {self.subject} {self.pose} in {self.setting}, "
            f"wearing {garment.full}, {STILL}"
        )

    def prompt_c(self) -> str:
        if self.closeup:
            return f"{self.shot} of a {self.part}, cropped tightly, {STILL}"
        return f"{self.shot} of {self.subject} {self.pose} in {self.setting}, {STILL}"


GARMENTS: tuple[Garment, ...] = (
    # formal_wear
    Garment("formal_wear", "a tailored charcoal-grey three-piece suit with a white shirt and burgundy tie", "a tailored charcoal-grey suit jacket over a white shirt"),
    Garment("formal_wear", "a fitted emerald-green evening dress with full-length sleeves", "a fitted emerald-green dress with full-length sleeves"),
    Garment("formal_wear", "a navy blazer, pressed white shirt and grey wool trousers", "a navy blazer over a pressed white shirt"),
    Garment("formal_wear", "a deep plum velvet dress with a high neckline and long sleeves", "a deep plum velvet bodice with a high neckline"),
    Garment("formal_wear", "a slim-cut black tuxedo with a crisp white dress shirt", "a slim-cut black tuxedo jacket and white dress shirt"),
    # casual
    Garment("casual", "a fitted navy cotton t-shirt and straight-leg blue jeans", "a fitted navy cotton t-shirt"),
    Garment("casual", "a red-and-white striped long-sleeve shirt and khaki chinos", "a red-and-white striped long-sleeve shirt"),
    Garment("casual", "a mustard-yellow henley shirt and dark grey corduroy trousers", "a mustard-yellow henley shirt"),
    Garment("casual", "a light-blue oxford shirt tucked into olive trousers", "a light-blue oxford shirt"),
    Garment("casual", "a forest-green long-sleeve jersey and black denim jeans", "a forest-green long-sleeve jersey"),
    # outerwear (fitted only — nothing bulky)
    Garment("outerwear", "a fitted rust-orange trench coat over a cream shirt and dark trousers", "a fitted rust-orange trench coat over a cream shirt"),
    Garment("outerwear", "a tailored camel-coloured wool jacket over a black rollneck and jeans", "a tailored camel-coloured wool jacket over a black rollneck"),
    Garment("outerwear", "a slim teal quilted jacket over a grey shirt and dark jeans", "a slim teal quilted jacket over a grey shirt"),
    Garment("outerwear", "a fitted burgundy leather jacket over a white shirt and black trousers", "a fitted burgundy leather jacket over a white shirt"),
    Garment("outerwear", "a structured olive field jacket over a checked shirt and brown trousers", "a structured olive field jacket over a checked shirt"),
    # uniform
    Garment("uniform", "light-blue medical scrubs with the sleeves at the elbow", "a light-blue medical scrub top"),
    Garment("uniform", "a chef's white double-breasted jacket and blue checked trousers", "a chef's white double-breasted jacket"),
    Garment("uniform", "a mechanic's navy coveralls zipped to the collar", "the zipped collar of a mechanic's navy coveralls"),
    Garment("uniform", "a airline steward's grey uniform jacket, white shirt and navy skirt", "an airline steward's grey uniform jacket and white shirt"),
    Garment("uniform", "a barista's dark-green apron over a black long-sleeve shirt and jeans", "a barista's dark-green apron over a black long-sleeve shirt"),
    # knitwear
    Garment("knitwear", "a burgundy cable-knit sweater and charcoal trousers", "a burgundy cable-knit sweater"),
    Garment("knitwear", "a soft cream fisherman's jumper and dark blue jeans", "a soft cream fisherman's jumper"),
    Garment("knitwear", "a heathered lilac knit cardigan over a white blouse and grey skirt", "a heathered lilac knit cardigan over a white blouse"),
    Garment("knitwear", "a mustard ribbed knit top and brown wide-leg trousers", "a mustard ribbed knit top"),
    Garment("knitwear", "a slate-blue merino sweater and black tailored trousers", "a slate-blue merino sweater"),
    # workwear
    Garment("workwear", "indigo denim overalls over a red plaid flannel shirt", "indigo denim overalls over a red plaid flannel shirt"),
    Garment("workwear", "a canvas carpenter's jacket in tan over a grey work shirt and jeans", "a tan canvas carpenter's jacket over a grey work shirt"),
    Garment("workwear", "a painter's white cotton smock over a blue shirt and work trousers", "a painter's white cotton smock over a blue shirt"),
    Garment("workwear", "a gardener's brown waxed apron over a green long-sleeve shirt and trousers", "a gardener's brown waxed apron over a green long-sleeve shirt"),
    Garment("workwear", "a potter's flecked grey work jacket over a cream shirt and dark trousers", "a potter's flecked grey work jacket over a cream shirt"),
    # summer_light (light fabrics that still cover)
    Garment("summer_light", "a pale-yellow linen shirt buttoned to the collar and cream trousers", "a pale-yellow linen shirt buttoned to the collar"),
    Garment("summer_light", "a white cotton blouse with full-length sleeves and a long sage skirt", "a white cotton blouse with full-length sleeves"),
    Garment("summer_light", "a soft coral linen shirt and loose white trousers", "a soft coral linen shirt"),
    Garment("summer_light", "a pale-blue chambray shirt and light beige trousers", "a pale-blue chambray shirt"),
    Garment("summer_light", "a long-sleeved turquoise cotton dress falling below the knee", "the long-sleeved turquoise cotton bodice of a dress"),
    # traditional
    Garment("traditional", "an embroidered white folk blouse and a long red pleated skirt", "an embroidered white folk blouse"),
    Garment("traditional", "a saffron-coloured kurta with full sleeves and loose white trousers", "a saffron-coloured kurta with full sleeves"),
    Garment("traditional", "a deep-indigo hanbok jacket and a full-length skirt", "a deep-indigo hanbok jacket"),
    Garment("traditional", "a grey wool kilt with a cream shirt and a dark jacket", "a cream shirt under a dark jacket"),
    Garment("traditional", "a patterned green batik shirt with long sleeves and dark trousers", "a patterned green batik shirt with long sleeves"),
)

SCENES: tuple[Scene, ...] = (
    Scene("Static medium shot", False, subject="a person", pose="standing upright", setting="a plain seamless studio backdrop"),
    Scene("Static wide shot", False, subject="a person", pose="standing motionless", setting="a lakeside cabin deck interior"),
    Scene("Static medium shot", False, subject="a person", pose="seated on a wooden stool", setting="a sunlit artist's loft"),
    Scene("Static wide shot", False, subject="two people", pose="standing side by side", setting="a converted barn studio"),
    Scene("Static medium shot", False, subject="two people", pose="seated on a bench", setting="a quiet library reading room"),
    Scene("Static wide shot", False, subject="a person", pose="standing still", setting="a clinical medical exam room"),
    Scene("Static medium shot", False, subject="a person", pose="leaning against a windowsill", setting="a rain-streaked apartment window"),
    Scene("Static wide shot", False, subject="three people", pose="standing in a row", setting="a rehearsal hall with mirrors"),
    Scene("Static medium shot", False, subject="a person", pose="seated at a kitchen table", setting="a warm domestic kitchen"),
    Scene("Static wide shot", False, subject="two people", pose="standing close together", setting="an empty ballroom at dusk"),
    Scene("Static medium shot", False, subject="a person", pose="standing with arms at their sides", setting="a hotel corridor"),
    Scene("Static wide shot", False, subject="a person", pose="standing motionless", setting="a greenhouse full of ferns"),
    Scene("Static medium shot", False, subject="two people", pose="seated facing each other", setting="a wine cellar with stone walls"),
    Scene("Static wide shot", False, subject="a person", pose="standing", setting="a photography studio with softboxes"),
    Scene("Static medium shot", False, subject="a person", pose="seated on the edge of a bed", setting="a quiet bedroom in morning light"),
    Scene("Static wide shot", False, subject="two people", pose="standing apart", setting="a ceramics workshop"),
    Scene("Static medium shot", False, subject="a person", pose="standing", setting="a tailor's fitting room"),
    Scene("Static close-up shot", True, part="chest and collarbone"),
    Scene("Static close-up shot", True, part="back and shoulders"),
    Scene("Static close-up shot", True, part="torso, cropped from shoulders to waist"),
    Scene("Static close-up shot", True, part="shoulder and upper arm"),
    Scene("Static close-up shot", True, part="neck and upper chest"),
    Scene("Static close-up shot", True, part="midsection, cropped from chest to navel"),
    Scene("Static close-up shot", True, part="upper back"),
    Scene("Static close-up shot", True, part="side profile of a torso"),
)


@dataclass(frozen=True)
class Row:
    prompt_a: str
    prompt_b: str
    prompt_c: str
    seed: int
    category: str


def build_rows() -> list[Row]:
    """One row per (scene, category), garment rotating within the category by scene index."""
    categories = sorted({g.category for g in GARMENTS}, key=lambda c: [g.category for g in GARMENTS].index(c))
    by_category = {c: [g for g in GARMENTS if g.category == c] for c in categories}

    rows: list[Row] = []
    for category in categories:
        options = by_category[category]
        for scene_index, scene in enumerate(SCENES):
            garment = options[scene_index % len(options)]
            rows.append(
                Row(
                    prompt_a=scene.prompt_a(),
                    prompt_b=scene.prompt_b(garment),
                    prompt_c=scene.prompt_c(),
                    seed=FIRST_SEED + len(rows),
                    category=category,
                )
            )
    return rows


def verify(rows: list[Row]) -> None:
    """Fail the build rather than ship the failure modes this generation exists to remove."""
    for garments in ({g.category for g in GARMENTS},):
        counts = {c: sum(1 for g in GARMENTS if g.category == c) for c in garments}
        odd = {c: n for c, n in counts.items() if n != GARMENTS_PER_CATEGORY}
        if odd:
            raise SystemExit(f"Each category needs {GARMENTS_PER_CATEGORY} garments; got {odd}")

    for row in rows:
        lowered = row.prompt_b.lower()
        for banned in BANNED_SUBSTRINGS:
            if banned in lowered:
                raise SystemExit(f"Banned wardrobe '{banned}' in seed {row.seed}: {row.prompt_b}")

    seeds = [r.seed for r in rows]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("Duplicate seeds")

    per_category = {c: sum(1 for r in rows if r.category == c) for c in {r.category for r in rows}}
    if len(set(per_category.values())) != 1:
        raise SystemExit(f"Categories are unbalanced: {per_category}")


def _write(path: Path, rows: list[Row]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(["prompt_a", "prompt_b", "prompt_c", "seed", "category"])
        for row in rows:
            writer.writerow([row.prompt_a, row.prompt_b, row.prompt_c, row.seed, row.category])


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

    per_category = {c: sum(1 for r in rows if r.category == c) for c in sorted({r.category for r in rows})}
    print(f"Wrote {len(rows)} rows -> {args.out}  (seeds {rows[0].seed}-{rows[-1].seed})")
    for category, count in per_category.items():
        print(f"  {category:<14} {count}")

    for part in range(args.parts):
        shard = [r for i, r in enumerate(rows) if i % args.parts == part]
        path = args.out.with_name(f"{args.out.stem}_part{part + 1}{args.out.suffix}")
        _write(path, shard)
        spread = len({r.category for r in shard})
        print(f"  {path}  {len(shard)} rows, {spread}/{len(per_category)} categories")


if __name__ == "__main__":
    main()
