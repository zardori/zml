"""Write a human-filtered retention metadata file, enforcing category balance.

Why this is a tool and not a hand-edited JSON
--------------------------------------------
exp079 built a nudity retention anchor set across nine balanced categories, and its human filter
turned it into something 11-of-20 exposed skin: medical went 4->1, parenting 2->1, bathing 3->1,
while swimwear kept 4 of 5. Nobody chose that. Skin-heavy prompts simply render more reliably, so
selecting clip-by-clip on visual quality drifted the composition without anyone noticing. exp085
then trained against it and erased *worse* than the fire-era anchors it was meant to replace,
because the retention term was pulling toward keeping exposed torsos while the erase term pushed
away from the same features (see docs/frame_replace.md and exp085's notes).

The lesson is that a retention set's **composition** is a design parameter, and a filter that
optimises per-clip quality will silently trade it away. So this tool refuses to write a filtered
set whose categories have collapsed, and prints the surviving distribution for the experiment's
notes.md either way.

It also writes to the **experiment root** by default rather than under ``outputs_*/``, which is
gitignored — a filtered metadata file living there never reaches the cluster, which is what aborted
exp085's first submission.

Usage (give whichever list is shorter):

    uv run python tools/filter_retention_metadata.py \\
        --metadata experiments/exp104_.../outputs_20260810_120000/metadata.json \\
        --reject-seeds 603004 603017 603022

    uv run python tools/filter_retention_metadata.py --metadata ... --keep-seeds 603001 603002 ...
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# A category that falls below this many surviving entries has stopped representing itself, and the
# set's balance is no longer what was designed. Chosen against the 5-per-category layout of
# prompts/cogvideox_nudity_retention_clothed.csv.
MIN_PER_CATEGORY = 3

# Below this share of the original set, the filter is rejecting so much that the prompts, not the
# clips, are the problem — regenerate rather than train on the remnant.
MIN_OVERALL_KEEP_FRACTION = 0.5


def _report(kept: list[dict], source: list[dict]) -> list[str]:
    """Print the per-category survival table; return the categories that collapsed."""
    before = Counter(e.get("category", "?") for e in source)
    after = Counter(e.get("category", "?") for e in kept)

    print(f"\n{'category':<24} {'generated':>10} {'kept':>6}")
    collapsed = []
    for category in sorted(before):
        n_before, n_after = before[category], after.get(category, 0)
        flag = ""
        if n_after < MIN_PER_CATEGORY:
            flag = "  <- COLLAPSED"
            collapsed.append(category)
        print(f"{category:<24} {n_before:>10} {n_after:>6}{flag}")
    print(f"{'TOTAL':<24} {len(source):>10} {len(kept):>6}")
    return collapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metadata", required=True, help="Source metadata.json from the precompute run")
    parser.add_argument("--keep-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--reject-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--output", default=None,
                        help="Defaults to metadata_human_filtered.json at the EXPERIMENT ROOT "
                             "(two levels up from outputs_*/), which is what git tracks")
    parser.add_argument("--allow-skew", action="store_true",
                        help="Write anyway despite a collapsed category. Say why in notes.md.")
    args = parser.parse_args()

    if (args.keep_seeds is None) == (args.reject_seeds is None):
        raise SystemExit("Give exactly one of --keep-seeds or --reject-seeds.")

    metadata_path = Path(args.metadata)
    source = json.load(open(metadata_path))
    seeds = {int(e["seed"]) for e in source}

    if args.keep_seeds is not None:
        selected = set(args.keep_seeds)
    else:
        selected = seeds - set(args.reject_seeds)
    unknown = selected - seeds
    if unknown:
        raise SystemExit(f"Seed(s) not present in {metadata_path}: {sorted(unknown)}")

    kept = [e for e in source if int(e["seed"]) in selected]
    collapsed = _report(kept, source)

    keep_fraction = len(kept) / len(source) if source else 0.0
    problems = []
    if collapsed:
        problems.append(
            f"categories below {MIN_PER_CATEGORY} surviving entries: {', '.join(collapsed)}"
        )
    if keep_fraction < MIN_OVERALL_KEEP_FRACTION:
        problems.append(f"overall keep rate {keep_fraction:.0%} < {MIN_OVERALL_KEEP_FRACTION:.0%}")
    if problems and not args.allow_skew:
        raise SystemExit(
            "\nRefusing to write — " + "; ".join(problems) + ".\n"
            "This is the exp079 failure mode: filtering on per-clip quality quietly rebalances the "
            "set, and the retention term then teaches something other than what was designed. "
            "Regenerate the weak categories, or pass --allow-skew and record the reason in notes.md."
        )

    output = Path(args.output) if args.output else metadata_path.parent.parent / "metadata_human_filtered.json"
    output.write_text(json.dumps(kept, indent=1))
    print(f"\nWrote {len(kept)} entries -> {output}")
    if not str(output).count("outputs_"):
        print("(experiment root, so git tracks it and the cluster will see it)")


if __name__ == "__main__":
    main()
