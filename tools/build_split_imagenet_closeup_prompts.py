#!/usr/bin/env python3
"""Rebuild the ImageNet split-prompt A/B/C CSVs with object-dominant framing.

Why
---
exp066/exp067 run 2 (construction-derived masks, ``split_step_frac: 0.85``) screened at 7/30 and
3/30 usable. ``tools/screen_split_dataset.py`` attributes the dominant loss to the same cause in
both classes: **17 of 30 clips never rendered the target object at all**, anywhere in the clip. That
is not a sampler failure — the splitter cannot separate a concept the base model did not draw.

The face thread hit this first and measured the fix. exp115 kept 9/30; cross-referencing showed 14
of the 21 rejects had ``original_max_confidence`` at or near 0, in wide/side-on/occluded framings.
exp116 rewrote the prompts with controlled medium/close frontal framing, held everything else fixed,
and yield went 30% -> 50% and 63% on the two reframed CSVs (a re-seed of the original prompts
reproduced 30% exactly, so it was framing, not seed luck).

Two things are wrong with the current object prompts, both visible against
``prompts/imagenet_objects.csv`` — the eval set the base model scores 0.739 (church) and 0.506
(chain saw) top-1 on:

1. **Framing.** "Static wide shot of a small church across a field of wildflowers" puts a small
   building in a large landscape. ResNet-50 classifies a 224px crop of the whole frame, so a small
   object is not merely hard to detect — it is genuinely not what the frame is *of*. The eval prompts
   put the object front and centre; the training prompts did not.
2. **Under-specification.** Eval prompts name the class-identifying parts ("with a tall steeple",
   "its orange casing and bar clearly visible"); the split prompts said only "a church" / "a chain
   saw". Those details are most of why the eval clips render as the class at all.

The fix applies both, and applies (2) **symmetrically to prompt B**. If only A gains detail, B is the
weaker prompt and loses the splice on prompt strength rather than on content — which would buy yield
by quietly turning the safe half into the concept half.

Held fixed on purpose: the 30 settings and their seed order (so every row is directly comparable to
exp066/exp067 run 2, same seed = same scene), the substitute objects, and the static-camera scaffold
— exp099 tested motion-carrying prompts and they were strictly worse (0/5 two-state against 2/5).

Run:  uv run python tools/build_split_imagenet_closeup_prompts.py
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

CHAIN_SAW_SOURCE = "prompts/imagenet_objects/split/chain_saw.csv"
CHURCH_SOURCE = "prompts/imagenet_objects/split/church.csv"
CHAIN_SAW_CSV = "prompts/imagenet_objects/split/chain_saw_closeup.csv"
CHURCH_CSV = "prompts/imagenet_objects/split/church_closeup.csv"
SWEEP_CSV = "prompts/imagenet_objects/split/chain_saw_closeup_sweep.csv"

# A second generation of the same 30 scenes under fresh seeds. exp117/exp118 measured these prompts
# at 14/30 usable per class, which is not enough rows to train on twice, and the cheapest way to more
# is more samples of a distribution we have now measured rather than more prompt editing. Re-seeding
# is also the control exp116 ran for faces: re-seeding *bad* prompts reproduced their yield exactly,
# so re-seeding good ones should reproduce 47% and the two generations can simply be merged.
CHAIN_SAW_GEN2_CSV = "prompts/imagenet_objects/split/chain_saw_closeup_gen2.csv"
CHURCH_GEN2_CSV = "prompts/imagenet_objects/split/church_closeup_gen2.csv"
# +30 keeps each generation in its own contiguous block (chain saw 3201-3230 then 3231-3260) so a
# seed still identifies its generation on sight, and no seed is ever reused across the thread.
GEN2_SEED_OFFSET = 30

# The 5 rows cheap sampler sweeps run on. One class only, because a sweep is scored with
# `screen_split_dataset.py` and a detector build takes a single `concept_target`.
#
# These are the five exp066 run-2 seeds whose peak p(chain saw) was highest (0.61-0.81), *not* the
# five that split best. A sampler sweep should ask "does this knob preserve a concept the model
# already renders", and on rows picked for rendering that question has a clean answer; on rows picked
# at random it is confounded with the 17-in-30 chance the object was never drawn at all.
SWEEP_SEEDS = (3202, 3204, 3208, 3210, 3217)

# The static-camera scaffold, unchanged from the CSVs these replace.
SUFFIX = "photorealistic. The camera is fixed and never moves."


@dataclass(frozen=True)
class Scene:
    """One row: a shared setting, the concept object, its substitute, and the row's seed.

    ``setting`` is the prepositional phrase A and B share. Prompt C is *not* rebuilt here — it is
    copied verbatim from the CSV this one replaces (keyed by seed), so the only thing that changes
    between exp066/exp067 run 2 and the runs built from this file is the framing and specificity of
    A and B. C is also the one prompt the tail phase sees, and exp099 measured that phase to be
    near-inert for content, so rewriting it would add a variable that buys nothing.
    """

    seed: int
    setting: str  # shared by A and B
    concept: str  # prompt A's subject, with its class-identifying detail
    substitute: str  # prompt B's subject, specified to the same level of detail


# "in close view, filling much of the frame" is the object-dominance clause; the concept and
# substitute phrases carry the identifying detail that the eval prompts have and the old split
# prompts lacked.
CHAIN_SAW_FRAMING = "in close view, filling much of the frame"
CHAIN_SAW = "a chain saw with an orange casing and a toothed guide bar"

CHAIN_SAW_SCENES: list[Scene] = [
    Scene(3201, "on a wooden workbench in a garage", CHAIN_SAW, "a chrome bicycle pump with a black rubber hose"),
    Scene(3202, "on a tree stump in a pine forest", CHAIN_SAW, "a green metal watering can with a long spout"),
    Scene(3203, "on the tailgate of a pickup truck at the edge of a wood", CHAIN_SAW, "an open paint can with a wire handle"),
    Scene(3204, "on the concrete floor of a workshop", CHAIN_SAW, "a claw hammer with a wooden handle"),
    Scene(3205, "on a canvas tarp in a timber yard", CHAIN_SAW, "a red steel crowbar"),
    Scene(3206, "on a pile of split firewood in a snowy yard", CHAIN_SAW, "a coil of thick hemp rope"),
    Scene(3207, "on a dark studio surface under a single soft light", CHAIN_SAW, "a blue plastic bucket"),
    Scene(3208, "in the bed of a trailer among cut branches", CHAIN_SAW, "a stainless steel thermos flask"),
    Scene(3209, "against a wooden shed wall", CHAIN_SAW, "a garden spade with a worn wooden shaft"),
    Scene(3210, "on a felled trunk in a forest clearing", CHAIN_SAW, "a red metal toolbox with a folding handle"),
    Scene(3211, "on a garage shelf between paint tins", CHAIN_SAW, "a copper oil can with a long thin spout"),
    Scene(3212, "on a rubber mat in a repair shop", CHAIN_SAW, "a black car battery with red and blue terminals"),
    Scene(3213, "on grass beside a hedge", CHAIN_SAW, "a galvanised watering can with a riveted handle"),
    Scene(3214, "on a stone wall in a rural garden", CHAIN_SAW, "a terracotta flowerpot"),
    Scene(3215, "on a workbench lit by a single work lamp", CHAIN_SAW, "a yellow cordless drill with a battery pack"),
    Scene(3216, "against a woodpile under a corrugated roof", CHAIN_SAW, "a stiff-bristled yard broom"),
    Scene(3217, "on the sawdust-covered floor of a sawmill", CHAIN_SAW, "a green metal jerry can"),
    Scene(3218, "on the back step of a forestry cabin", CHAIN_SAW, "a pair of green rubber boots"),
    Scene(3219, "on a gravel drive beside a log pile", CHAIN_SAW, "a steel wheelbarrow with a pneumatic tyre"),
    Scene(3220, "on a flatbed cart in an orchard", CHAIN_SAW, "a woven wicker basket"),
    Scene(3221, "on a plastic crate at the edge of a field", CHAIN_SAW, "a folded blue tarpaulin"),
    Scene(3222, "on a wooden pallet in a builder's yard", CHAIN_SAW, "a paper cement bag"),
    Scene(3223, "on a picnic table in a campsite clearing", CHAIN_SAW, "a small camping stove with a brass burner"),
    Scene(3224, "on metal shelving in a storage unit", CHAIN_SAW, "a plain cardboard box"),
    Scene(3225, "on the porch of a log cabin", CHAIN_SAW, "a blue enamel kettle"),
    Scene(3226, "on a tarpaulin at a roadside clearing site", CHAIN_SAW, "an orange and white traffic cone"),
    Scene(3227, "on a bench in a garden shed hung with tools", CHAIN_SAW, "a battered zinc watering can"),
    Scene(3228, "on a crate beside a chopping block", CHAIN_SAW, "a pair of leather work gloves"),
    Scene(3229, "on the floor of an open barn doorway", CHAIN_SAW, "a bale of straw bound with twine"),
    Scene(3230, "on a workbench beside a metal vice", CHAIN_SAW, "a roll of silver duct tape"),
]

# Church is a building, so the dominance clause is phrased for architecture rather than a close-up.
# The concept phrase varies its identifying feature (steeple / spire / bell tower) the way the eval
# prompts do; the substitutes deliberately have *no* tower, spire or bell-cote, so the "concept-free"
# half is genuinely church-free to the classifier.
CHURCH_FRAMING = "the building filling most of the frame"
STEEPLE = "a stone church with a tall pointed steeple"
SPIRE = "a church with a slender spire"
BELL_TOWER = "a church with a square bell tower"
GOTHIC = "a gothic church with a tall tower and arched stained-glass windows"

CHURCH_SCENES: list[Scene] = [
    Scene(3301, "in a green English village on a clear afternoon", STEEPLE, "a stone barn with a long slate roof"),
    Scene(3302, "across a field of wildflowers in summer", SPIRE, "a low farmhouse with a slate roof"),
    Scene(3303, "on a quiet street corner in low evening sun", BELL_TOWER, "a flat-roofed brick warehouse"),
    Scene(3304, "on a hillside overlooking the sea", BELL_TOWER, "a whitewashed cottage with a flat tiled roof"),
    Scene(3305, "across a cobbled square", GOTHIC, "a plain townhouse facade with rows of sash windows"),
    Scene(3306, "among pine trees under a grey sky", STEEPLE, "a low wooden lodge with a shallow roof"),
    Scene(3307, "beyond an old graveyard at dawn", BELL_TOWER, "a squat stone gatehouse"),
    Scene(3308, "above the rooftops of a small town", SPIRE, "a cylindrical grain silo"),
    Scene(3309, "in a meadow at golden hour", STEEPLE, "a long cattle shed with an open front"),
    Scene(3310, "in a Mediterranean village at midday", BELL_TOWER, "a flat-fronted house with a red-tiled roof"),
    Scene(3311, "in a snowy winter landscape", STEEPLE, "a timber cabin with a low snow-covered roof"),
    Scene(3312, "at the end of a tree-lined avenue in autumn", SPIRE, "a broad stone manor house with tall chimneys"),
    Scene(3313, "beside a narrow country road with dry stone walls", BELL_TOWER, "a small stone barn with a corrugated roof"),
    Scene(3314, "against a stormy sky in dramatic light", GOTHIC, "a steel water tower on lattice legs"),
    Scene(3315, "alone on a wide prairie under an open sky", STEEPLE, "a long clapboard farm building"),
    Scene(3316, "lit by floodlights at night above an empty square", GOTHIC, "a colonnaded museum facade"),
    Scene(3317, "with sunlight breaking through surrounding trees", SPIRE, "a wooden boathouse with wide doors"),
    Scene(3318, "reflected in a still pond in front of it", STEEPLE, "a brick mill house with a water wheel"),
    Scene(3319, "on a hilltop with stone steps climbing toward it", BELL_TOWER, "a hilltop farmhouse with a low slate roof"),
    Scene(3320, "in a fishing village above grey harbour water", SPIRE, "a black-tarred net store with a shallow roof"),
    Scene(3321, "behind an iron fence on an overcast morning", GOTHIC, "a two-storey school building with wide windows"),
    Scene(3322, "across a frozen lake in pale winter light", STEEPLE, "a timber hunting lodge with a broad porch"),
    Scene(3323, "at the edge of a village green", BELL_TOWER, "a single-storey village hall with a flat roof"),
    Scene(3324, "on a terraced hillside above vineyards", SPIRE, "a long winery building with arched cellar doors"),
    Scene(3325, "seen through an archway in a stone wall", BELL_TOWER, "a courtyard house with shuttered windows"),
    Scene(3326, "beside a canal on a still grey morning", STEEPLE, "a brick workshop with tall metal windows"),
    Scene(3327, "surrounded by bare winter trees", SPIRE, "a small cottage with a thatched roof"),
    Scene(3328, "at the top of a sloping cobbled street", BELL_TOWER, "a bakery with a wide shopfront window"),
    Scene(3329, "beyond a field of grazing sheep", STEEPLE, "an open-sided hay barn"),
    Scene(3330, "on a coastal headland under a wide sky", GOTHIC, "a low white coastguard station"),
]


def load_prompt_c(source_csv: Path, scenes: list[Scene]) -> dict[int, str]:
    """Prompt C per seed, taken verbatim from the CSV being replaced (see ``Scene``)."""
    with source_csv.open(newline="") as handle:
        by_seed = {int(row["seed"]): row["prompt_c"] for row in csv.DictReader(handle)}
    missing = [s.seed for s in scenes if s.seed not in by_seed]
    if missing:
        raise SystemExit(f"{source_csv} has no row for seed(s) {missing}; scene table and source disagree.")
    return by_seed


def build_row(scene: Scene, framing: str, prompt_c: str) -> dict[str, str | int]:
    """The A/B/C triple for one scene."""
    return {
        "prompt_a": f"Static shot of {scene.concept} {scene.setting}, {framing}, {SUFFIX}",
        "prompt_b": f"Static shot of {scene.substitute} {scene.setting}, {framing}, {SUFFIX}",
        "prompt_c": prompt_c,
        "seed": scene.seed,
    }


def build_rows(source_csv: Path, scenes: list[Scene], framing: str) -> list[dict[str, str | int]]:
    prompt_c = load_prompt_c(source_csv, scenes)
    return [build_row(s, framing, prompt_c[s.seed]) for s in scenes]


def reseed(rows: list[dict[str, str | int]], offset: int) -> list[dict[str, str | int]]:
    """The same triples under a fresh seed block — one more sample of a measured yield (see GEN2)."""
    return [{**row, "seed": int(row["seed"]) + offset} for row in rows]


def write_csv(path: Path, rows: list[dict[str, str | int]], note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt_a", "prompt_b", "prompt_c", "seed"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path}: {len(rows)} triples ({note})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chain-saw-csv", default=CHAIN_SAW_CSV)
    parser.add_argument("--church-csv", default=CHURCH_CSV)
    parser.add_argument("--sweep-csv", default=SWEEP_CSV)
    parser.add_argument("--chain-saw-gen2-csv", default=CHAIN_SAW_GEN2_CSV)
    parser.add_argument("--church-gen2-csv", default=CHURCH_GEN2_CSV)
    args = parser.parse_args()

    chain_saw = build_rows(Path(CHAIN_SAW_SOURCE), CHAIN_SAW_SCENES, CHAIN_SAW_FRAMING)
    church = build_rows(Path(CHURCH_SOURCE), CHURCH_SCENES, CHURCH_FRAMING)
    write_csv(Path(args.chain_saw_csv), chain_saw, f"seeds {CHAIN_SAW_SCENES[0].seed}-{CHAIN_SAW_SCENES[-1].seed}")
    write_csv(Path(args.church_csv), church, f"seeds {CHURCH_SCENES[0].seed}-{CHURCH_SCENES[-1].seed}")

    for path, rows in ((args.chain_saw_gen2_csv, chain_saw), (args.church_gen2_csv, church)):
        gen2 = reseed(rows, GEN2_SEED_OFFSET)
        write_csv(Path(path), gen2, f"gen2 re-seed, seeds {gen2[0]['seed']}-{gen2[-1]['seed']}")

    by_seed = {int(row["seed"]): row for row in church + chain_saw}
    missing = [s for s in SWEEP_SEEDS if s not in by_seed]
    if missing:
        raise SystemExit(f"sweep seed(s) {missing} are in neither class CSV.")
    write_csv(Path(args.sweep_csv), [by_seed[s] for s in SWEEP_SEEDS],
              f"subset of the two class CSVs, seeds {list(SWEEP_SEEDS)}")


if __name__ == "__main__":
    main()
