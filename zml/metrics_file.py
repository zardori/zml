"""Safe concurrent updates to a run's ``metrics.json``.

Several post-hoc scorers write into the *same* ``eval_step_*/metrics.json``, each owning a different
subset of fields: ``tools/score_dover.py`` (DOVER), ``tools/score_nudity_frame_rate.py`` (the
frame-level nudity rate), ``tools/score_q16.py`` (Q16 and the OR rate). Each one reads the file,
merges its own keys, and writes the whole thing back.

Run two of them at once and the second write silently discards the first one's fields — the classic
lost update. That is not hypothetical: on 2026-08-10 a DOVER pass and a frame-rate pass were started
concurrently over exp084, and the DOVER write (which had read the file before the frame rates
landed) erased ``nudity_frame_rate`` from both of its runs. Nothing failed, nothing warned; the
fields were simply gone, and were noticed only because a later table build hit a ``None``.

``update_metrics_json`` closes that hole with an advisory lock held across the read-merge-write.
Scoring stays *outside* the lock — callers compute their numbers first and pass in the finished
updates — so the critical section is a few milliseconds of file I/O and concurrent scorers still
run in parallel where the expensive work is.
"""

import fcntl
import json
from pathlib import Path


def update_metrics_json(path: Path, updates: dict[str, dict]) -> dict:
    """Merge ``updates`` into ``path`` under an exclusive lock; return the merged contents.

    ``updates`` maps a prompt-set name (``concept``, ``related``, ``unrelated``, ...) to the fields
    that set gains. Existing fields the caller does not mention are preserved, so two scorers owning
    disjoint keys can both land even if they run back to back.

    A missing file is treated as empty, which lets a scorer create metrics for a run whose eval job
    died before writing any (the ``tools/score_eval_videos.py`` recovery case).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open r+ when possible so the lock guards the read as well as the write; fall back to creating.
    with open(path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            raw = handle.read()
            metrics = json.loads(raw) if raw.strip() else {}

            for set_name, fields in updates.items():
                existing = metrics.get(set_name)
                if isinstance(existing, dict):
                    existing.update(fields)
                else:
                    metrics[set_name] = dict(fields)

            handle.seek(0)
            handle.truncate()
            json.dump(metrics, handle, indent=2)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return metrics
