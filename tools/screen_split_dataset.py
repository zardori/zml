"""Select the split-prompt clips that actually carry erase signal, from the confidences precompute
already logged. Concept-agnostic: works on any dataset built by ``frame_replace_split_precompute``.

Why a *differential* test
-------------------------
The obvious screen — "did the detector fire?" — is what cost exp066/exp067 their first run. An
absolute threshold asks ``p(church) > 0.03?`` on a scale that is not comparable across scenes: over
exp067's 30 clips the per-clip maximum p(church) ranges from 0.0009 to 0.357, driven mostly by
framing and lighting. Any single cut through that range is arbitrary, and the run that tried one
produced masks that disagreed with the construction on six of seven kept rows.

The question that *is* well posed is a within-clip, paired one: **does the half conditioned on the
concept prompt read more concept than the half conditioned on the safe prompt?** Both halves share a
seed, a scene, a camera and a lighting setup, so everything except the concept cancels. That is what
this tool measures, as a bounded contrast index

    ci = (mean(concept_half) - mean(safe_half)) / (mean(concept_half) + mean(safe_half))

in [-1, 1]: +1 means the concept lives entirely in its own half, 0 means the split did nothing, and
negative means the safe half reads *more* concept than the concept half (it happens — 16 of exp067's
30 rows).

A contrast index alone is not enough, because a clip can split cleanly into two states neither of
which contains the target. exp066's ``p13_s3214`` is the case in point: the first half is a flower
pot, the second a chain-and-hook object, the seam is textbook — and the peak p(chain saw) over all 49
frames is 0.003. So a row must clear **both** gates: the concept half must actually contain the
concept (``--min-concept-max``), and it must be separated from the safe half (``--min-contrast-index``).

Relation to the other two checkers
----------------------------------
``tools/check_seam_contrast.py`` measures the same property in pixel space and is concept-blind, so
it makes exactly the two mistakes above: it passes ``p13_s3214`` (two states, no chain saw) and fails
``p27_s3328`` (a correct church split whose bell tower is too small to move a whole-frame mean).
Prefer this tool to decide what to train on, and keep seam contrast for diagnosing *why* a clip
failed. ``tools/screen_split_face_dataset.py`` is the absolute-threshold-only ancestor of this file,
still in place because exp115/exp116's published keep-lists were selected with it.

Thresholds are deliberately CLI arguments rather than a per-concept table: ``build_detector`` is the
one place the codebase maps a concept string to behaviour, and this tool must not become a second
one. Calibrations measured on exp064's base-model reference, to pass explicitly:

    chain saw   --min-concept-max 0.10    (genuine clips median 0.115; other classes never above 0.044)
    church      --min-concept-max 0.10    (genuine clips median 0.233; other classes never above 0.006)

    uv run python tools/screen_split_dataset.py --metadata <outputs_.../metadata.json>
"""

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

# CogVideoX packs 49 pixel frames into 13 latent frames: latent frame 0 covers a single pixel frame
# and every later one covers 4, so latent frame `sf` starts at pixel frame 1 + 4*(sf-1).
PIXEL_FRAMES_PER_LATENT = 4

# Peak detector score the concept half must reach for prompt A to count as having rendered the
# concept at all. Detector-specific — see the calibration table in the module docstring.
MIN_CONCEPT_MAX = 0.10
# How separated the two halves must be. 0.4 sits inside a clear gap in the church data (the three
# genuine splits score 0.87/0.63/0.49, the next row down scores 0.18) and keeps every clip that
# survived visual review in both classes.
MIN_CONTRAST_INDEX = 0.4


@dataclass(frozen=True)
class ScreenResult:
    """One clip's concept-separation diagnosis."""

    stem: str
    seed: int
    split_latent_frame: int
    concept_region: str
    concept_max: float
    concept_mean: float
    safe_mean: float
    contrast_index: float
    verdict: str


def seam_pixel_frame(split_latent_frame: int) -> int:
    """First pixel frame belonging to the second temporal region."""
    return 1 + PIXEL_FRAMES_PER_LATENT * (split_latent_frame - 1)


def split_halves(entry: dict) -> tuple[list[float], list[float]]:
    """``(concept_half, safe_half)`` per-frame confidences, ordered by construction not position."""
    confidences = entry["frame_confidences"]
    seam = seam_pixel_frame(entry["split_latent_frame"])
    first, second = confidences[:seam], confidences[seam:]
    return (first, second) if entry["concept_region"] == "first" else (second, first)


def contrast_index(concept_half: list[float], safe_half: list[float]) -> float:
    """Bounded [-1, 1] separation between the two halves; 0 when the split changed nothing."""
    concept_mean, safe_mean = statistics.mean(concept_half), statistics.mean(safe_half)
    total = concept_mean + safe_mean
    return (concept_mean - safe_mean) / total if total > 0 else 0.0


def screen_entry(entry: dict, min_concept_max: float, min_contrast: float) -> ScreenResult:
    concept_half, safe_half = split_halves(entry)
    concept_max = max(concept_half)
    index = contrast_index(concept_half, safe_half)
    if concept_max < min_concept_max:
        verdict = "no-concept"  # prompt A never rendered it; nothing to erase
    elif index < min_contrast:
        verdict = "not-split"  # concept is there, but the safe half has it too
    else:
        verdict = "pass"
    return ScreenResult(
        stem=Path(entry["latent_path"]).name.removesuffix("_x0edited.pt"),
        seed=int(entry["seed"]),
        split_latent_frame=entry["split_latent_frame"],
        concept_region=entry["concept_region"],
        concept_max=concept_max,
        concept_mean=statistics.mean(concept_half),
        safe_mean=statistics.mean(safe_half),
        contrast_index=index,
        verdict=verdict,
    )


def print_results(results: list[ScreenResult]) -> None:
    print(f"{'stem':14} {'sf':>3} {'region':>7} {'conc_max':>9} {'conc_mean':>10} "
          f"{'safe_mean':>10} {'contrast':>9}  verdict")
    for r in results:
        print(f"{r.stem:14} {r.split_latent_frame:3d} {r.concept_region:>7} {r.concept_max:9.4f} "
              f"{r.concept_mean:10.4f} {r.safe_mean:10.4f} {r.contrast_index:+9.3f}  {r.verdict}")


def region_balance(results: list[ScreenResult]) -> dict[str, int]:
    """`concept_region` counts among survivors — a skewed keep set teaches the positional shortcut."""
    return {region: sum(1 for r in results if r.concept_region == region) for region in ("first", "second")}


def write_filtered(metadata_path: Path, entries: list[dict], survivors: list[ScreenResult],
                   destination: Path | None) -> Path:
    """Write the surviving entries to the *experiment root*, which — unlike ``outputs_*/`` — is not
    gitignored, so the filtered set actually reaches the cluster (the mistake that aborted exp085)."""
    keep_seeds = {r.seed for r in survivors}
    out = destination or metadata_path.parent.parent / f"{metadata_path.parent.name}_screened.json"
    out.write_text(json.dumps([e for e in entries if int(e["seed"]) in keep_seeds], indent=2))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True, help="metadata.json from frame_replace_split_precompute")
    parser.add_argument("--min-concept-max", type=float, default=MIN_CONCEPT_MAX,
                        help="Peak detector score the concept half must reach (detector-specific)")
    parser.add_argument("--min-contrast-index", type=float, default=MIN_CONTRAST_INDEX,
                        help="Minimum [-1,1] separation between the concept and safe halves")
    parser.add_argument("--write-filtered", nargs="?", const="", default=None, metavar="PATH",
                        help="Write surviving entries as JSON (default: <experiment>/<outputs>_screened.json)")
    parser.add_argument("--min-pass-frac", type=float, default=None,
                        help="Exit non-zero if the pass fraction falls below this")
    parser.add_argument("--limit", type=int, default=None, help="Screen only the first N entries")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    entries = json.load(open(metadata_path))[: args.limit]
    if not entries:
        raise SystemExit(f"{metadata_path} has no entries.")

    results = [screen_entry(e, args.min_concept_max, args.min_contrast_index) for e in entries]
    print_results(results)

    counts = {v: sum(1 for r in results if r.verdict == v) for v in ("pass", "not-split", "no-concept")}
    survivors = [r for r in results if r.verdict == "pass"]
    pass_frac = len(survivors) / len(results)
    print(f"\n{len(results)} clips | pass {counts['pass']} ({pass_frac:.0%}) | "
          f"not-split {counts['not-split']} | no-concept {counts['no-concept']}")
    if counts["no-concept"]:
        print(f"{counts['no-concept']} clips never rendered the concept at all — a prompt/framing "
              f"problem, not a sampler one (see exp116: controlling framing nearly doubled face yield).")

    if survivors:
        balance = region_balance(survivors)
        print(f"surviving concept_region balance: {balance['first']} first / {balance['second']} second")
        print(f"--keep-seeds {' '.join(str(r.seed) for r in survivors)}")
        if args.write_filtered is not None:
            destination = Path(args.write_filtered) if args.write_filtered else None
            print(f"wrote {len(survivors)} entries -> {write_filtered(metadata_path, entries, survivors, destination)}")

    if args.min_pass_frac is not None and pass_frac < args.min_pass_frac:
        print(f"FAIL: pass fraction {pass_frac:.0%} is below --min-pass-frac {args.min_pass_frac:.0%}.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
