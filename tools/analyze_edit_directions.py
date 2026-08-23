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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True, help="Filtered metadata.json naming the seeds actually trained on")
    parser.add_argument("--videos", required=True, action="append", help="Directory of *_original.mp4 / *_edited.mp4 (repeatable)")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    seeds = {int(e["seed"]) for e in json.load(open(args.metadata))}
    pairs: dict[int, tuple[Path, Path]] = {}
    for d in args.videos:
        for edited in Path(d).rglob("*_edited.mp4"):
            m = re.search(r"_s(\d+)_edited\.mp4$", edited.name)
            if not m:
                continue
            seed = int(m.group(1))
            original = edited.with_name(edited.name.replace("_edited.mp4", "_original.mp4"))
            if seed in seeds and original.exists():
                pairs.setdefault(seed, (original, edited))

    edits = [e for s, (o, ed) in sorted(pairs.items()) if (e := measure(s, o, ed))]
    if not edits:
        raise SystemExit("No original/edited pairs matched the metadata seeds.")

    dirs = np.stack([e.direction for e in edits])
    norms = np.linalg.norm(dirs, axis=1)
    coherence = float(np.linalg.norm(dirs.mean(axis=0)) / norms.mean())
    unit = dirs / np.clip(norms[:, None], 1e-8, None)
    pairwise = float((unit @ unit.T)[np.triu_indices(len(unit), 1)].mean()) if len(unit) > 1 else float("nan")

    label = args.label or Path(args.metadata).parent.name
    print(f"{label}: matched {len(edits)}/{len(seeds)} seeds")
    print(f"  edit magnitude (mean LAB L2)      {np.mean([e.magnitude for e in edits]):8.3f}")
    print(f"  COHERENCE ||mean d|| / mean||d||  {coherence:8.3f}   (1 = all edits align, 0 = cancel)")
    print(f"  mean pairwise cosine              {pairwise:8.3f}")
    print(f"  mean LAB shift (L, a, b)          {np.array2string(dirs.mean(axis=0), precision=2)}")


if __name__ == "__main__":
    main()
