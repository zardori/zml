"""Measure WHY one frame_replace dataset erases better than another.

Two properties of a training set determine how much erase signal a low-rank adapter can extract,
and neither is visible in the per-clip metrics we normally look at:

1. **Edit magnitude** — how far the grafted target sits from the original. The erase push is
   ``eta * (donor - teacher)``, so a dataset whose donors barely differ from the concept clip
   delivers a small displacement at fixed eta. This is why gen4's fitted wardrobe needed higher eta
   than gen1-gen3's baggy sacks.

2. **Edit COHERENCE across the dataset** — the decisive one, and the reason this tool exists. A
   rank-r LoRA can only realise the components of the edit that *recur across examples*: it learns
   roughly the shared direction of ``(donor - teacher)``, not each example's private one. If every
   clip's edit points the same way ("replace skin with dark heavy fabric"), that shared direction is
   strong and generalizes. If the edits point in many directions — which is exactly what deliberate
   wardrobe diversity produces — they partially cancel in the average, the shared component shrinks
   toward *style* rather than *clothing*, and the adapter learns something that does not erase.

   Quantified as ``coherence = ||mean(d_i)|| / mean(||d_i||)`` over per-clip edit vectors ``d_i``.
   1.0 = every edit points identically; 0.0 = they cancel completely.

Edits are summarised **scene-invariantly** as the mean LAB colour shift over edited frames, so
clips of different scenes are comparable — a spatial delta would be dominated by scene layout and
say nothing about whether the *kind* of change is shared.

Run:
    uv run python tools/analyze_edit_directions.py --metadata <filtered.json> --videos <dir> [--videos <dir>]
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Frames are decimated before colour statistics: the LAB mean is a global statistic, so full
# resolution buys nothing and costs decode time.
RESIZE = (64, 64)


@dataclass(frozen=True)
class ClipEdit:
    seed: int
    magnitude: float          # mean per-pixel L2 in LAB between edited and original
    direction: np.ndarray     # mean LAB shift, a 3-vector


def _lab_frames(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, RESIZE, interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32))
    cap.release()
    return np.stack(out) if out else np.empty((0, *RESIZE, 3), np.float32)


def measure(seed: int, original: Path, edited: Path) -> ClipEdit | None:
    a, b = _lab_frames(original), _lab_frames(edited)
    n = min(len(a), len(b))
    if n == 0:
        return None
    delta = b[:n] - a[:n]                       # (F, H, W, 3)
    magnitude = float(np.linalg.norm(delta, axis=-1).mean())
    direction = delta.reshape(-1, 3).mean(axis=0)
    return ClipEdit(seed, magnitude, direction)


@dataclass(frozen=True)
class Summary:
    n: int
    magnitude: float
    coherence: float
    pairwise: float
    shift: np.ndarray         # mean LAB shift (L, a, b)

    @property
    def luma(self) -> float:
        """|dL| of the shared direction — a global brightness change, semantically empty."""
        return float(abs(self.shift[0]))

    @property
    def chroma(self) -> float:
        """||(da, db)|| of the shared direction — the colour-family change, where skin lives."""
        return float(np.linalg.norm(self.shift[1:]))

    @property
    def shared(self) -> float:
        """||mean d|| — the size of the component a low-rank adapter can actually learn.
        Read this *before* the ratio: chroma:luma explodes when luma -> 0 and is only
        meaningful next to the absolute chroma it is a ratio of."""
        return float(np.linalg.norm(self.shift))

    @property
    def chroma_luma(self) -> float:
        """The decisive ratio. >1 means the learnable edit is 'change colour of skin regions';
        <1 means it is mostly 'make the frame darker', which a LoRA applies to everything."""
        return self.chroma / max(self.luma, 1e-8)


def summarize(edits: list[ClipEdit]) -> Summary:
    dirs = np.stack([e.direction for e in edits])
    norms = np.linalg.norm(dirs, axis=1)
    unit = dirs / np.clip(norms[:, None], 1e-8, None)
    pairwise = float((unit @ unit.T)[np.triu_indices(len(unit), 1)].mean()) if len(unit) > 1 else float("nan")
    return Summary(
        n=len(edits),
        magnitude=float(np.mean([e.magnitude for e in edits])),
        coherence=float(np.linalg.norm(dirs.mean(axis=0)) / norms.mean()),
        pairwise=pairwise,
        shift=dirs.mean(axis=0),
    )


def print_summary(label: str, s: Summary, total: int | None = None) -> None:
    matched = f"{s.n}/{total}" if total is not None else str(s.n)
    print(f"{label}: matched {matched} seeds")
    print(f"  edit magnitude (mean LAB L2)      {s.magnitude:8.3f}")
    print(f"  COHERENCE ||mean d|| / mean||d||  {s.coherence:8.3f}   (1 = all edits align, 0 = cancel)")
    print(f"  mean pairwise cosine              {s.pairwise:8.3f}")
    print(f"  mean LAB shift (L, a, b)          {np.array2string(s.shift, precision=2)}")
    print(f"  shared component ||mean d||       {s.shared:8.2f}   (what a LoRA can learn)")
    print(f"  luma |dL| / chroma ||da,db||      {s.luma:8.2f} / {s.chroma:.2f}")
    print(f"  CHROMA:LUMA                       {s.chroma_luma:8.2f}   "
          f"({'concept-like' if s.chroma_luma > 1 else 'style-like'})")


def load_pairs(video_dirs: list[str], seeds: set[int]) -> dict[int, tuple[Path, Path]]:
    """Resolve each metadata seed to exactly one (original, edited) pair.

    Both failure modes here silently corrupted a result once and must stay fatal:

    * **Partial match.** A subset of the seeds is not the dataset. Measuring 13 of 34 clips of the
      exp080 set gave chroma:luma 0.40 on one subset and 1.47 on another — opposite conclusions
      from the same dataset, and the only warning was a "matched 13/34" line.
    * **Ambiguous match.** A precompute *grid* writes the same seeds under every ``run_00N/``, one
      per hyperparameter value. Passing several of them offers several different edits per seed,
      and picking one by directory order silently measures a build nobody trained on. Pass only the
      run the dataset was actually merged from (the experiment's ``notes.md`` names it).
    """
    found: dict[int, list[tuple[Path, Path]]] = {}
    for d in video_dirs:
        for edited in Path(d).rglob("*_edited.mp4"):
            m = re.search(r"_s(\d+)_edited\.mp4$", edited.name)
            if not m:
                continue
            seed = int(m.group(1))
            original = edited.with_name(edited.name.replace("_edited.mp4", "_original.mp4"))
            if seed in seeds and original.exists():
                found.setdefault(seed, []).append((original, edited))

    if missing := sorted(seeds - set(found)):
        raise SystemExit(
            f"{len(missing)}/{len(seeds)} metadata seeds have no video pair in --videos "
            f"(e.g. {missing[:8]}). A subset is not the dataset; supply every source dir."
        )
    if ambiguous := {s: v for s, v in found.items() if len(v) > 1}:
        seed, variants = next(iter(sorted(ambiguous.items())))
        dirs = "\n  ".join(str(e.parent) for _, e in variants)
        raise SystemExit(
            f"{len(ambiguous)} seeds match in more than one --videos dir (seed {seed} in "
            f"{len(variants)}):\n  {dirs}\nThese are usually precompute-grid variants of the same "
            f"prompts. Pass only the run this dataset was merged from."
        )
    return {s: v[0] for s, v in found.items()}


def load_groups(path: str, column: str) -> dict[int, str]:
    with open(path, newline="") as f:
        return {int(row["seed"]): row[column] for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True, help="Filtered metadata.json naming the seeds actually trained on")
    parser.add_argument("--videos", required=True, action="append", help="Directory of *_original.mp4 / *_edited.mp4 (repeatable)")
    parser.add_argument("--label", default=None)
    parser.add_argument("--groups", default=None,
                        help="CSV with a `seed` column, to also report each subset separately "
                             "(e.g. the build's prompt CSV). This is how a candidate dataset is "
                             "screened before training: a subset with chroma:luma > 1 is what to build more of.")
    parser.add_argument("--group-column", default="category", help="Column of --groups to partition on")
    parser.add_argument("--min-group", type=int, default=4, help="Skip subsets smaller than this")
    args = parser.parse_args()

    seeds = {int(e["seed"]) for e in json.load(open(args.metadata))}
    pairs = load_pairs(args.videos, seeds)
    edits = [e for s, (o, ed) in sorted(pairs.items()) if (e := measure(s, o, ed))]
    if not edits:
        raise SystemExit("No original/edited pairs matched the metadata seeds.")

    label = args.label or Path(args.metadata).parent.name
    print_summary(label, summarize(edits), total=len(seeds))  # equal by construction

    if args.groups:
        groups = load_groups(args.groups, args.group_column)
        buckets: dict[str, list[ClipEdit]] = {}
        for e in edits:
            if (g := groups.get(e.seed)) is not None:
                buckets.setdefault(g, []).append(e)
        summaries = {g: summarize(v) for g, v in buckets.items() if len(v) >= args.min_group}
        print(f"\n  by {args.group_column} (n >= {args.min_group}), sorted by chroma:luma:")
        print(f"    {'group':<24} {'n':>3} {'mag':>7} {'coher':>6} {'shared':>7} {'luma':>6} {'chroma':>7} {'c:l':>6}")
        for g, s in sorted(summaries.items(), key=lambda kv: -kv[1].chroma_luma):
            print(f"    {g:<24} {s.n:>3} {s.magnitude:>7.2f} {s.coherence:>6.3f} {s.shared:>7.2f} "
                  f"{s.luma:>6.2f} {s.chroma:>7.2f} {s.chroma_luma:>6.2f}")


if __name__ == "__main__":
    main()
