"""Pre-screen a split-prompt face dataset's ArcFace confidences, so human review only watches the
clips with a chance of passing.

Why this exists
----------------
exp115 (Barack Obama, ``split_step_frac: 0.8``) kept 9 of 30 triples on human review. Cross-
referencing the keep list against ``metadata.json`` afterwards showed the dominant loss was not the
split sampler: 14 of the 21 rejects had ``original_max_confidence`` at or near 0.0 — the base model
simply never rendered a recognizable Obama for that (prompt, seed) at all, in a wide/side-on/occluded
framing. Every one of the 9 keeps cleared both ``original_max_confidence`` and the whole-clip A-side
confidence comfortably above 0.3.

Both fields are already written by ``zml/precompute/frame_replace_split_precompute.py`` for every
row — a human doesn't need to watch a clip to know the detector found no face at all in it. This
tool reads them straight from ``metadata.json`` (no GPU, no new inference) and prints which entries
are worth watching.

This is a **triage, not a verdict**. ``docs/split_prompt.md`` documents that gating *inside*
precompute on the detector cost exp078 half its yield, which is why the precompute detector stays
logging-only there. This tool only decides which clips a human still has to watch by eye for splice
quality and whole-clip identity separation — the human keep list, written with
``tools/filter_retention_metadata.py``, remains the source of truth.

Calibrated against exp115's 30 triples: at the defaults below, all 9 human keeps pass and 14 of the
21 rejects are cut, with zero false rejections (see this file's module docstring test in
``experiments/exp116_split_face_obama_dataset_scaleup/notes.md``).

    uv run python tools/screen_split_face_dataset.py --metadata <outputs_.../metadata.json>
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# ArcFace cosine similarity, same scale as IDENTITY_THRESHOLD (0.23, zml/benchmarks/check_for_face.py).
# Set with margin above it: exp115's 9 keeps all clear 0.344/0.323, its 21 rejects split cleanly with
# 14 at ~0.000 and the rest scattered up to 0.546 (still rejected on other grounds), so this only
# screens out the "no face rendered at all" failures, not the borderline ones a human should still see.
MIN_ORIGINAL_MAX_CONFIDENCE = 0.30
MIN_WHOLECLIP_A_MAX_CONFIDENCE = 0.30


@dataclass(frozen=True)
class ScreenResult:
    seed: int
    stem: str
    original_max_confidence: float
    wholeclip_a_max_confidence: float
    passed: bool


def video_stem(entry: dict) -> str:
    """The `p{row}_s{seed}` stem, matching tools/check_seam_contrast.py's convention."""
    return Path(entry["latent_path"]).name.removesuffix("_x0edited.pt")


def screen_entry(entry: dict, min_original: float, min_wholeclip_a: float) -> ScreenResult:
    original_max = float(entry["original_max_confidence"])
    wholeclip = entry.get("variants", {}).get("wholeclip")
    wholeclip_a_max = max(wholeclip["frame_confidences"]) if wholeclip else 0.0
    passed = original_max >= min_original and wholeclip_a_max >= min_wholeclip_a
    return ScreenResult(
        seed=int(entry["seed"]),
        stem=video_stem(entry),
        original_max_confidence=original_max,
        wholeclip_a_max_confidence=wholeclip_a_max,
        passed=passed,
    )


def print_results(results: list[ScreenResult]) -> None:
    header = f"{'stem':16} {'seed':>6} {'orig_max':>9} {'wc_a_max':>9}  verdict"
    print(header)
    for r in results:
        verdict = "PASS" if r.passed else "cut"
        print(f"{r.stem:16} {r.seed:6d} {r.original_max_confidence:9.3f} "
              f"{r.wholeclip_a_max_confidence:9.3f}  {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True, help="metadata.json from frame_replace_split_precompute")
    parser.add_argument("--min-original-max", type=float, default=MIN_ORIGINAL_MAX_CONFIDENCE)
    parser.add_argument("--min-wholeclip-a-max", type=float, default=MIN_WHOLECLIP_A_MAX_CONFIDENCE)
    parser.add_argument("--limit", type=int, default=None, help="Screen only the first N entries")
    args = parser.parse_args()

    entries = json.load(open(args.metadata))[: args.limit]
    if not entries:
        raise SystemExit(f"{args.metadata} has no entries.")
    missing_wholeclip = [e["seed"] for e in entries if "wholeclip" not in e.get("variants", {})]
    if missing_wholeclip:
        raise SystemExit(
            f"{len(missing_wholeclip)} entries have no 'variants.wholeclip' (was "
            f"emit_whole_clip_target enabled?): seeds {missing_wholeclip}"
        )

    results = [screen_entry(e, args.min_original_max, args.min_wholeclip_a_max) for e in entries]
    print_results(results)

    survivors = [r for r in results if r.passed]
    print(f"\n{len(survivors)}/{len(results)} survive screening and are worth watching by eye.")
    if survivors:
        seeds = " ".join(str(r.seed) for r in survivors)
        print(f"--keep-seeds candidate list: {seeds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
