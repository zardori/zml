"""Report whether a split-prompt clip actually holds *two* states, in pixel space, from its MP4s.

Why this exists
---------------
A split-prompt training target only carries erase signal if the concept half and the safe half settle
on **different** content. The failure that costs us is a clip that collapses to one state across all
49 frames: prompt C's heal phase merges the two halves, both sides show the same thing, and
``x0_edited`` ends up near-identical to ``x0_original`` — a target that teaches nothing.

Absolute motion is the wrong way to detect this. A clip can be almost perfectly static within each
half and still be an excellent target, because the *step* at the seam is where the supervision lives.
exp067's ``p25_s3226`` has a median frame-to-frame difference of 0.47 — which reads as "static" — but
a 17.0 jump exactly at its seam, and it is one of the few good clips in that batch. Averaging over all
48 frame pairs buries that single transition.

So this measures **seam contrast** instead: the largest consecutive-frame difference, where it sits
relative to the seam the sampler was told to build, and how far it stands above the within-half noise.

- ``two-state``  — one dominant transition, at the seam. What we want.
- ``collapsed``  — no transition anywhere; the clip is one state. Useless as a target.
- ``diffuse``    — motion spread across the clip with no seam standing out. Either the halves never
                   separated, or there is so much motion that the boundary is smeared away
                   (exp067 ``p16_s3317``: median 17.6, max/median 1.2).

It works on the ``videos/*_original.mp4`` that ``pull_results.sh`` already downloads by default, so it
needs no GPU, no VAE and — unlike ``check_latent_motion.py`` — no ``.pt`` files off the cluster.
That one measures a different thing (is the *donor fill* frozen, in latent space); the two are
complements, not alternatives.

    uv run python tools/check_seam_contrast.py --metadata <meta.json> --videos-dir <dir>
"""

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# CogVideoX packs 49 pixel frames into 13 latent frames: latent frame 0 is a single pixel frame, and
# every latent frame after it covers 4. So latent frame `sf` — the first frame of the second region —
# begins at this pixel index, and the visual transition lands on the diff *before* it.
PIXEL_FRAMES_PER_LATENT = 4


def seam_diff_index(split_latent_frame: int) -> int:
    """Index into the consecutive-difference array where the split's transition should appear."""
    return PIXEL_FRAMES_PER_LATENT * (split_latent_frame - 1)


# The heal phase can drag the visible boundary by about a latent frame, so an exact hit is too strict;
# 2 latent frames still rules out a transition that landed in the wrong half entirely.
SEAM_TOLERANCE_FRAMES = 2 * PIXEL_FRAMES_PER_LATENT
# How far the seam must stand above the within-half median to count as a distinct state change.
MIN_SEAM_RATIO = 5.0
# Below this the largest "transition" in the clip is indistinguishable from encoder noise.
COLLAPSED_MAX_DIFF = 1.5


@dataclass(frozen=True)
class SeamReport:
    """Seam diagnostics for one clip."""

    stem: str
    seed: int
    split_latent_frame: int
    concept_region: str
    median_diff: float
    max_diff: float
    argmax: int
    seam: int
    ratio: float
    verdict: str


def consecutive_diffs(video_path: Path) -> np.ndarray:
    """Mean absolute grayscale difference between each consecutive pair of frames."""
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    finally:
        capture.release()
    if len(frames) < 2:
        raise ValueError(f"{video_path} has {len(frames)} frame(s); need at least 2.")
    return np.array([np.abs(b - a).mean() for a, b in zip(frames, frames[1:])])


def classify(median_diff: float, max_diff: float, argmax: int, seam: int, tolerance: int) -> tuple[float, str]:
    """Return (max/median ratio, verdict) for one clip's difference profile."""
    ratio = max_diff / median_diff if median_diff > 1e-6 else float("inf")
    if max_diff < COLLAPSED_MAX_DIFF:
        return ratio, "collapsed"
    if ratio >= MIN_SEAM_RATIO and abs(argmax - seam) <= tolerance:
        return ratio, "two-state"
    return ratio, "diffuse"


# The two producers name their clips differently: frame_replace_split_precompute writes
# `<stem>_original.mp4` next to the edited copy, while the split_prompt sampler writes
# `<stem>_combined.mp4` (plus optional plain A/B/C). Both are the same thing for our purposes — the
# manufactured partial-concept clip — so try them in order rather than making the caller care.
DEFAULT_CLIP_SUFFIXES = ("original", "combined")


def video_stem(entry: dict) -> str:
    """The `p{row}_s{seed}` stem, however this metadata flavour records it."""
    if "stem" in entry:
        return str(entry["stem"])
    return Path(entry["latent_path"]).name.removesuffix("_x0edited.pt")


def resolve_video(entry: dict, metadata_dir: Path, videos_dir: Path, suffix: str | None) -> Path:
    """Locate one row's clip across both metadata flavours."""
    stem = video_stem(entry)
    suffixes = (suffix,) if suffix else DEFAULT_CLIP_SUFFIXES
    candidates = []
    for candidate_suffix in suffixes:
        # The sampler records relative paths explicitly; the builder implies them from the stem.
        recorded = (entry.get("videos") or {}).get(candidate_suffix)
        if recorded:
            candidates.append(metadata_dir / recorded)
        candidates.append(videos_dir / f"{stem}_{candidate_suffix}.mp4")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no clip for {stem}; tried {[str(c) for c in candidates]}")


def report_clip(entry: dict, metadata_dir: Path, videos_dir: Path, suffix: str | None,
                tolerance: int) -> SeamReport:
    stem = video_stem(entry)
    diffs = consecutive_diffs(resolve_video(entry, metadata_dir, videos_dir, suffix))
    median_diff = float(np.median(diffs))
    max_diff = float(diffs.max())
    argmax = int(diffs.argmax())
    seam = seam_diff_index(entry["split_latent_frame"])
    ratio, verdict = classify(median_diff, max_diff, argmax, seam, tolerance)
    return SeamReport(
        stem=stem,
        seed=entry["seed"],
        split_latent_frame=entry["split_latent_frame"],
        concept_region=entry["concept_region"],
        median_diff=median_diff,
        max_diff=max_diff,
        argmax=argmax,
        seam=seam,
        ratio=ratio,
        verdict=verdict,
    )


def print_reports(reports: list[SeamReport]) -> None:
    header = f"{'stem':16} {'sf':>3} {'region':>7} {'median':>8} {'max':>8} {'@':>4} {'seam':>5} {'ratio':>7}  verdict"
    print(header)
    for r in reports:
        print(
            f"{r.stem:16} {r.split_latent_frame:3d} {r.concept_region:>7} {r.median_diff:8.3f} "
            f"{r.max_diff:8.2f} {r.argmax:4d} {r.seam:5d} {r.ratio:7.1f}  {r.verdict}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--videos-dir", help="Defaults to <metadata dir>/videos")
    parser.add_argument("--suffix", default=None,
                        help="Which MP4 per row, e.g. 'original' / 'combined' / 'edited'. "
                             f"Default: first of {DEFAULT_CLIP_SUFFIXES} that exists.")
    parser.add_argument("--seam-tolerance", type=int, default=SEAM_TOLERANCE_FRAMES,
                        help="Pixel frames the dominant transition may sit away from the seam")
    parser.add_argument("--min-two-state-frac", type=float, default=None,
                        help="Exit non-zero if the two-state fraction falls below this")
    parser.add_argument("--limit", type=int, default=None, help="Check only the first N entries")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    videos_dir = Path(args.videos_dir) if args.videos_dir else metadata_path.parent / "videos"
    entries = json.load(open(metadata_path))[: args.limit]

    reports = [report_clip(e, metadata_path.parent, videos_dir, args.suffix, args.seam_tolerance)
               for e in entries]
    print_reports(reports)

    counts = {v: sum(1 for r in reports if r.verdict == v) for v in ("two-state", "diffuse", "collapsed")}
    two_state_frac = counts["two-state"] / len(reports) if reports else 0.0
    median_ratio = statistics.median(r.ratio for r in reports) if reports else 0.0
    print(
        f"\n{len(reports)} clips | two-state {counts['two-state']} "
        f"({two_state_frac:.0%}) | diffuse {counts['diffuse']} | collapsed {counts['collapsed']} "
        f"| median seam ratio {median_ratio:.1f}"
    )
    if counts["collapsed"]:
        print("A collapsed clip shows the same content in both halves, so its target carries no erase signal.")

    if args.min_two_state_frac is not None and two_state_frac < args.min_two_state_frac:
        print(f"FAIL: two-state fraction {two_state_frac:.0%} is below --min-two-state-frac "
              f"{args.min_two_state_frac:.0%}.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
