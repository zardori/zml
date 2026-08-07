"""One ordering for the clips in an eval directory, shared by every scorer.

Scorers return *per-video* lists (``clip_scores``, ``motion_scores``, ...) that land side by side in
a run's ``metrics.json``, and ``score_video_dir`` pairs the i-th clip with the i-th prompt. All of
that is only meaningful if every scorer walks the directory in the same order — and until this
module existed they did not: ``VideoClipScorer`` sorted numerically (it has to, to line up with the
prompt list) while the detector, colorfulness, motion and DOVER scorers used a plain lexicographic
``sorted()``. With 115 clips, ``video_10`` sits at index 2 under one rule and index 10 under the
other; 113 of the 115 positions disagree.

Aggregate means and rates were never affected — they do not depend on order — but the per-video
arrays written for runs with ten or more clips were not mutually indexable.

The natural ordering below sorts embedded integers by value, so ``video_2`` precedes ``video_10``
and the numeric convention ``VideoClipScorer`` needs is preserved for ``video_{i}.mp4`` names, while
any other naming scheme still gets a total, deterministic order.
"""

import os
import re

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")


def _natural_key(name: str) -> list[tuple[int, int, str]]:
    """Split into digit and non-digit runs, comparing digit runs by value.

    Each element is a uniform 3-tuple so comparisons never mix ``int`` with ``str`` (which would
    raise) no matter how differently two filenames are shaped.
    """
    return [
        (1, int(part), "") if part.isdigit() else (0, 0, part)
        for part in re.split(r"(\d+)", name)
    ]


def list_video_files(video_dir: str) -> list[str]:
    """Video filenames in ``video_dir``, in the canonical order every scorer must use."""
    return sorted(
        (f for f in os.listdir(video_dir) if f.endswith(VIDEO_EXTENSIONS)),
        key=_natural_key,
    )
