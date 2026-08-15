"""Turning a run's ``.mp4`` output into something a browser will show.

Two jobs, both with a trap behind them.

**Frames.** Nothing in this project ever writes an image — every visual artifact is a 49-frame mp4 —
so a still has to be extracted before it can sit in a deck. Strips of evenly spaced frames are the
default because they scan, print and screenshot; clips are reserved for claims about *motion*, which
a still genuinely cannot carry.

**Codecs.** Clips written before commit ``e2c51a9`` are ``mpeg4``, which Chromium will not play — the
element renders as a black rectangle with no error. Every clip is probed and only re-encoded when it
has to be, so the common case stays a file copy.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from zml.video_files import list_video_files

MEDIA_DIR_NAME = "media"

FRAME_WIDTH = 360  # a four-frame strip then fits a slide's content column without scaling
JPEG_QUALITY = 82
DEFAULT_FRAMES = 4

BROWSER_CODEC = "h264"
TRANSCODE_ARGS = ("-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "23")

# Eval prompt sets, most-shown first: what was erased, then what had to survive it.
EVAL_GROUPS = ("concept", "related", "unrelated", "anchor")
# Precompute pairs the model's own output against the concept-removed edit built from it.
PAIR_SUFFIXES = ("_original.mp4", "_edited.mp4")


@dataclass(frozen=True)
class Candidate:
    """A clip on disk that could go in the deck, with the label it would carry."""

    path: Path
    label: str
    group: str  # prompt set, or "dataset" for a precompute build


def _probe_codec(video: Path) -> str | None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() or None


def eval_candidates(output_dir: Path, step: int | None) -> list[Candidate]:
    """Clips from one eval checkpoint, ordered so index *i* is still prompt row *i*."""
    if step is None:
        return []
    step_dir = output_dir / f"eval_step_{step}"
    return [
        Candidate(step_dir / group / name, f"{group} · {Path(name).stem}", group)
        for group in EVAL_GROUPS
        if (step_dir / group).is_dir()
        for name in list_video_files(str(step_dir / group))
    ]


def dataset_candidates(output_dir: Path) -> list[Candidate]:
    """Original/edited pairs from a precompute build, kept adjacent so they render as a comparison."""
    videos = output_dir / "videos"
    if not videos.is_dir():
        return []
    return [
        Candidate(videos / f"{stem}{suffix}", f"{stem} · {suffix[1:-4]}", "dataset")
        for stem in sorted({
            name[: -len(PAIR_SUFFIXES[0])]
            for name in list_video_files(str(videos))
            if name.endswith(PAIR_SUFFIXES[0])
        })
        for suffix in PAIR_SUFFIXES
        if (videos / f"{stem}{suffix}").exists()
    ]


def _write_frames(video: Path, out_dir: Path, stem: str, count: int) -> list[str]:
    capture = cv2.VideoCapture(str(video))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        # Evenly spaced over the clip, endpoints included — the seam a split-prompt build creates
        # sits mid-clip, so sampling only the head would hide the thing under review.
        indices = [round(i * (total - 1) / max(count - 1, 1)) for i in range(count)]

        written = []
        for position, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            height = round(frame.shape[0] * FRAME_WIDTH / frame.shape[1])
            resized = cv2.resize(frame, (FRAME_WIDTH, height), interpolation=cv2.INTER_AREA)
            name = f"{stem}_f{index:02d}.jpg"
            cv2.imwrite(str(out_dir / name), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            written.append(f"{MEDIA_DIR_NAME}/{name}")
        return written
    finally:
        capture.release()


def extract_strip(video: Path, media_dir: Path, stem: str, count: int = DEFAULT_FRAMES) -> list[str]:
    """Evenly spaced frames as deck-relative paths. Idempotent: existing frames are reused."""
    media_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(media_dir.glob(f"{stem}_f*.jpg"))
    if len(existing) == count:
        return [f"{MEDIA_DIR_NAME}/{path.name}" for path in existing]
    return _write_frames(video, media_dir, stem, count)


def stage_clip(video: Path, media_dir: Path, stem: str) -> str | None:
    """Copy a clip into the deck, re-encoding only if the browser could not play it."""
    media_dir.mkdir(parents=True, exist_ok=True)
    target = media_dir / f"{stem}.mp4"
    if target.exists():
        return f"{MEDIA_DIR_NAME}/{target.name}"

    if _probe_codec(video) == BROWSER_CODEC:
        shutil.copy2(video, target)
        return f"{MEDIA_DIR_NAME}/{target.name}"

    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), *TRANSCODE_ARGS, str(target)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        target.unlink(missing_ok=True)
        return None
    return f"{MEDIA_DIR_NAME}/{target.name}"


def media_bytes(media_dir: Path) -> int:
    return sum(path.stat().st_size for path in media_dir.glob("*") if path.is_file())
