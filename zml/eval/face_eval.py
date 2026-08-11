"""ID-Similarity evaluation for celebrity face/identity erasure (``docs/face_identity.md``).

The protocol comes from T2VUnlearning (arXiv 2505.17550 §4.3): erase one of five identities,
generate videos for *all* five, score every clip's ArcFace cosine similarity to the ground-truth
identity, and report

    Erase    = mean ID-similarity of the erased identity's own clips     (want low)
    Preserve = mean ID-similarity of the other four identities' clips    (want ~unchanged)

This does not fit ``zml/unlearn/eval.py::evaluate``, which scores one detector across
``concept``/``related``/``unrelated`` sets: here there are five sets, each scored against *its own*
target identity — the same shape as the ImageNet ESR/PSR problem, so this module mirrors
``imagenet_eval.py`` structurally throughout.

With ``erased_identity`` unset the run scores the unmodified base model and reports Erase/Preserve
for every identity in turn as the hypothetical erased one, plus mean and std across the five — this
is exactly how the paper's ``Original`` row and its ± are produced, from a single generation pass.

**Two ID-similarity conventions, both always computed** (``docs/face_identity.md`` §3.1): the
face-conditioned mean (headline — pooled only over frames where a face was actually detected) and
the zero-filled mean (auxiliary — a no-face frame counts as similarity 0), nested under
``"zerofill"``. ``face_present_rate`` accompanies every number; a low Erase score with a collapsed
face-presence rate is degradation, not erasure, and must not be read as a win.

Generation is resumable exactly like ``imagenet_eval.py``. A finished run can be re-scored without a
GPU pipeline (YuNet + ArcFace are ONNX-CPU)::

    uv run python -m zml.eval.face_eval --rescore experiments/expNNN_.../outputs_TS \\
        --prompts-csv prompts/face_cogvideox.csv [--skip-quality] [--cross-reference]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
import torch
import wandb
from diffusers.utils import export_to_video

from zml.benchmarks.arcface_embedder import ArcFaceFrameEmbedder
from zml.benchmarks.check_for_face import IDENTITY_THRESHOLD, VideoFaceDetector, read_bgr_frames
from zml.benchmarks.face_identities import FACE_IDENTITIES, identity_slug, load_all_reference_embeddings
from zml.benchmarks.frame_quality import degenerate_frame_mask
from zml.benchmarks.ort_runtime import DEFAULT_NUM_WORKERS
from zml.eval.clip_score import VideoClipScorer
from zml.eval.colorfulness import VideoColorfulnessScorer
from zml.eval.eval_model import build_eval_pipeline
from zml.eval.motion import VideoMotionScorer
from zml.video_files import list_video_files

# Standalone eval runs at a synthetic step so outputs share the eval_step_<n>/ layout.
EVAL_STEP = 0
VIDEO_FPS = 8
REQUIRED_COLUMNS = ("prompt", "seed", "class_name")
HEADLINE_METRICS = ("Erase", "Preserve")
# Key under which the zero-filled copy of the whole report is nested. The top level stays
# face-conditioned (the headline convention) so existing readers of id_similarity.json keep working
# if the convention is ever revisited (see docs/face_identity.md §3.1).
ZEROFILL_KEY = "zerofill"
# `negative_prompt: auto` expands to the erased identity's name — the NegPrompt baseline.
AUTO_NEGATIVE_PROMPT = "auto"


@dataclass
class Config:
    model_id: str
    output_dir: str
    eval_inference_steps: int
    prompts_csv: str  # columns: prompt, seed, class_name (+ optional case_number/source, ignored)
    # Which identity this model was trained to erase. None -> base model; Erase/Preserve is then
    # reported for every identity in turn, which fills the whole `Original` comparison row from one run.
    erased_identity: str | None = None
    lora_checkpoint_dir: str | None = None
    # Inference-time negative prompt. "auto" resolves to `erased_identity` (the NegPrompt baseline);
    # any other string is passed through verbatim.
    negative_prompt: str | None = None
    eval_num_prompts_per_identity: int | None = None  # None -> every row of each identity (30)
    identity_threshold: float = IDENTITY_THRESHOLD
    num_frames: int = 49
    guidance_scale: float = 6.0
    disable_mlflow: bool = False

    def __post_init__(self) -> None:
        if self.erased_identity is not None and self.erased_identity not in FACE_IDENTITIES:
            raise ValueError(
                f"erased_identity {self.erased_identity!r} is not one of {sorted(FACE_IDENTITIES)}."
            )
        if self.negative_prompt == AUTO_NEGATIVE_PROMPT and self.erased_identity is None:
            raise ValueError("negative_prompt: auto needs erased_identity to name what to negate.")

    def resolved_negative_prompt(self) -> str | None:
        if self.negative_prompt == AUTO_NEGATIVE_PROMPT:
            return self.erased_identity
        return self.negative_prompt


@dataclass
class IdentityScores:
    """Frame-pooled ID-similarity of every clip generated for one identity, under one convention.

    ``face_present_rate``/``identified_rate`` must be read alongside ``id_sim`` — see
    ``docs/face_identity.md`` §3.1's hard reporting rule.
    """

    id_sim: float
    id_sim_zerofill: float
    face_present_rate: float
    identified_rate: float  # face_detection_rate: fraction of clips whose id_sim >= identity_threshold
    num_videos: int
    clips_without_face: int
    # Clips with >=1 blank/structureless frame (generation failure, not a real "no face"
    # measurement -- see zml/benchmarks/frame_quality.py) and the frame-pooled rate across this
    # identity's clips. face_present_rate/id_sim above already exclude degenerate frames from their
    # denominators; these two fields are what makes that exclusion auditable in the report.
    clips_degenerate: int = 0
    degenerate_frame_rate: float = 0.0
    # Mean pairwise cosine among this identity's own per-clip mean face embeddings; None if fewer
    # than 2 clips have a detected face. High values mean the model's renders of this identity are
    # converging on one specific face regardless of prompt/seed — the "fixed-substitute collapse"
    # failure mode (docs/face_identity.md §5): erasure that reads as success only because it always
    # draws the same replacement face, not because it removed the identity generally.
    collapse_score: float | None = None

    def zerofill(self) -> "IdentityScores":
        """This identity's zero-filled score in the plain ``id_sim`` slot, so the Erase/Preserve
        maths is shared between conventions (mirrors ``imagenet_eval.ClassScores.restricted``)."""
        return IdentityScores(
            id_sim=self.id_sim_zerofill, id_sim_zerofill=self.id_sim_zerofill,
            face_present_rate=self.face_present_rate, identified_rate=self.identified_rate,
            num_videos=self.num_videos, clips_without_face=self.clips_without_face,
            clips_degenerate=self.clips_degenerate, degenerate_frame_rate=self.degenerate_frame_rate,
            collapse_score=self.collapse_score,
        )


def load_identity_prompts(csv_path: str, limit: int | None = None) -> dict[str, list[tuple[str, int]]]:
    """Group a prompt CSV into ``{identity_name: [(prompt, seed), ...]}``, in file order.

    Seeds come from the file per the repo seed policy. For ``prompts/face_cogvideox.csv`` these are
    T2VUnlearning's own published seeds, not ours (``tools/fetch_face_eval_prompts.py``).
    """
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s) {sorted(missing)}.")

    unknown = set(df["class_name"]) - set(FACE_IDENTITIES)
    if unknown:
        raise ValueError(f"{csv_path} has class_name values outside the protocol: {sorted(unknown)}.")

    grouped: dict[str, list[tuple[str, int]]] = {}
    for name, rows in df.groupby("class_name", sort=False):
        pairs = [(str(p), int(s)) for p, s in zip(rows["prompt"], rows["seed"])]
        grouped[str(name)] = pairs[:limit] if limit else pairs
    return grouped


def identity_video_dir(eval_root: str, identity_name: str) -> str:
    """Where one identity's clips live. Shared by generation and scoring so they cannot drift apart."""
    return os.path.join(eval_root, identity_slug(identity_name))


def _video_needs_regeneration(video_path: str) -> bool:
    """True if ``video_path`` is missing, empty, or decodes to an entirely degenerate clip.

    Content-aware resume predicate. The old ``getsize(video_path) > 0`` check treated any non-empty
    file as already generated, so a black clip (~3.4 KB, well clear of empty) was silently skipped
    forever on every resumed run -- see ``zml/benchmarks/frame_quality.py``. A *partially* degenerate
    clip is left alone: it still carries real content once degenerate frames are excluded from
    ``check_for_face``'s denominators, so only a fully blank clip counts as missing.
    """
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return True
    frames = read_bgr_frames(video_path)
    return not frames or all(degenerate_frame_mask(frames))


def _generate_identity_videos(pipe, config: Config, identity_name: str, prompts: list[tuple[str, int]],
                              eval_root: str) -> str:
    video_dir = identity_video_dir(eval_root, identity_name)
    os.makedirs(video_dir, exist_ok=True)
    negative_prompt = config.resolved_negative_prompt()

    with torch.no_grad():
        for i, (prompt, seed) in enumerate(prompts):
            video_path = os.path.join(video_dir, f"video_{i}.mp4")
            if not _video_needs_regeneration(video_path):
                continue  # resume a killed job without paying for it again
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=config.num_frames,
                num_inference_steps=config.eval_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=torch.Generator(device=pipe.device).manual_seed(seed),
            )
            export_to_video(result.frames[0], video_path, fps=VIDEO_FPS)
            print(f"Saved eval video: {video_path}")
    return video_dir


def compute_erase_preserve(per_identity: dict[str, IdentityScores], erased_identity: str) -> dict[str, float]:
    """Erase/Preserve ID-similarity for one choice of erased identity."""
    if erased_identity not in per_identity:
        raise ValueError(f"No scores for erased identity {erased_identity!r}; have {sorted(per_identity)}.")
    others = [s for name, s in per_identity.items() if name != erased_identity]
    if not others:
        raise ValueError("Preserve needs at least one non-erased identity.")
    return {
        "Erase": per_identity[erased_identity].id_sim,
        "Preserve": float(np.mean([s.id_sim for s in others])),
    }


def _leave_one_out_report(per_identity: dict[str, IdentityScores]) -> dict:
    """Erase/Preserve with each identity in turn as the erased one, plus mean/std across the five.

    This is what the published ``Original`` row reports: the ± spread comes from varying which
    identity counts as erased, not from repeated sampling. For a base-model run, mean Erase = mean
    Preserve = mean Original — all three collapse to the overall mean id-similarity by construction
    (mirrors ``imagenet_eval``'s "ESR-1 + PSR-1 = 100" note).
    """
    rows = {name: compute_erase_preserve(per_identity, name) for name in per_identity}
    return {
        "per_erased_identity": rows,
        "mean": {m: float(np.mean([r[m] for r in rows.values()])) for m in HEADLINE_METRICS},
        "std": {m: float(np.std([r[m] for r in rows.values()])) for m in HEADLINE_METRICS},
    }


def _scores_block(per_identity: dict[str, IdentityScores], erased_identity: str | None) -> dict:
    """Per-identity scores plus the Erase/Preserve summary, under whichever convention holds."""
    block: dict = {
        "per_identity": {
            name: {
                "id_sim": round(s.id_sim, 4),
                "face_present_rate": round(s.face_present_rate, 4),
                "identified_rate": round(s.identified_rate, 4),
                "num_videos": s.num_videos,
                "clips_without_face": s.clips_without_face,
                "clips_degenerate": s.clips_degenerate,
                "degenerate_frame_rate": round(s.degenerate_frame_rate, 4),
                "collapse_score": round(s.collapse_score, 4) if s.collapse_score is not None else None,
            }
            for name, s in per_identity.items()
        }
    }
    if erased_identity is not None:
        block.update(compute_erase_preserve(per_identity, erased_identity))
    else:
        block.update(_leave_one_out_report(per_identity))
    return block


def _headline(block: dict, erased_identity: str | None) -> dict[str, float]:
    """The two numbers a table row is made of, however the block was built."""
    return {m: block[m] for m in HEADLINE_METRICS} if erased_identity is not None else block["mean"]


def _clip_degenerate_flags(video_dir: str) -> list[bool]:
    """Per-clip "has >=1 blank/structureless frame" flags, in ``list_video_files`` order.

    A third pass over the same clips, like ``_collapse_score`` below -- but plain pixel-std
    (``zml/benchmarks/frame_quality.py``), not ArcFace/YuNet, so it costs a fraction of even that
    CPU-cheap pass. Kept separate rather than threading through any scorer's ``list[float]``
    interface, same reasoning as ``_collapse_score``.
    """
    flags = []
    for name in list_video_files(video_dir):
        frames = read_bgr_frames(os.path.join(video_dir, name))
        flags.append(not frames or any(degenerate_frame_mask(frames)))
    return flags


def _quality_scores(video_dir: str, prompts: list[str], degenerate_flags: list[bool] | None = None) -> dict[str, float]:
    """Generic quality signals per identity, so a collapse in Erase can be read against video quality.

    Degenerate clips (``degenerate_flags``, see ``_clip_degenerate_flags``) are excluded before
    averaging: a black clip reads as motion/colorfulness == 0, which would otherwise read as a
    genuine quality collapse rather than a generation failure unrelated to what is being evaluated.
    """
    clip = VideoClipScorer(video_dir=video_dir, prompts=prompts).process_videos()
    color = VideoColorfulnessScorer(video_dir=video_dir).process_videos()
    motion = VideoMotionScorer(video_dir=video_dir).process_videos()

    def _valid(scores: list[float]) -> list[float]:
        if not degenerate_flags:
            return scores
        return [s for s, degenerate in zip(scores, degenerate_flags) if not degenerate]

    clip, color, motion = _valid(clip), _valid(color), _valid(motion)
    return {
        "clip_score_mean": float(np.mean(clip)) if clip else 0.0,
        "colorfulness_mean": float(np.mean(color)) if color else 0.0,
        "motion_score_mean": float(np.mean(motion)) if motion else 0.0,
    }


def _collapse_score(detector: VideoFaceDetector, video_dir: str) -> float | None:
    """Mean pairwise cosine among one identity's own per-clip mean face embeddings.

    A second scoring pass over the same clips (``process_videos()`` already scored them once) — kept
    separate rather than threading embeddings through the plain ``dict[str, float]`` interface every
    other detector in ``zml/benchmarks/`` returns. Face detection is CPU-cheap relative to the GPU
    generation cost this evaluates, so the duplicated pass is a simplicity trade worth making.
    """
    video_files = list_video_files(video_dir)
    embeddings = []
    for name in video_files:
        stats = detector.score_video(os.path.join(video_dir, name))
        if stats.clip_embedding is not None:
            embeddings.append(stats.clip_embedding)
    if len(embeddings) < 2:
        return None
    stacked = np.stack(embeddings)
    sims = stacked @ stacked.T
    n = len(embeddings)
    return float((sims.sum() - np.trace(sims)) / (n * (n - 1)))


def _cross_reference_one_clip(video_path: str) -> np.ndarray:
    """One clip's face-conditioned mean similarity against every reference -- a ``(5,)`` vector.

    Module-level (not a method) so it can run in a worker process: pickling a live
    ``ArcFaceFrameEmbedder`` (it holds ONNX sessions) doesn't work, so each worker builds its own via
    ``_init_cross_reference_worker``, exactly the ``_WORKER_DETECTOR`` pattern
    ``check_for_face.py``'s ``_score_one`` already uses.
    """
    assert _CROSS_REF_WORKER_EMBEDDER is not None and _CROSS_REF_WORKER_MATRIX is not None, (
        "cross-reference worker not initialised"
    )
    frames = read_bgr_frames(video_path)
    face_sims: list[np.ndarray] = []  # one (5,) row per face-bearing, non-degenerate frame
    if frames:
        degenerate = degenerate_frame_mask(frames)
        per_frame_faces = _CROSS_REF_WORKER_EMBEDDER.embed_frames(frames)
        for faces, is_degenerate in zip(per_frame_faces, degenerate):
            if is_degenerate or len(faces) == 0:
                continue
            sims = faces.embeddings @ _CROSS_REF_WORKER_MATRIX.T  # (num_faces_in_frame, 5)
            face_sims.append(sims.max(axis=0))
    return np.mean(face_sims, axis=0) if face_sims else np.zeros(_CROSS_REF_WORKER_MATRIX.shape[0])


_CROSS_REF_WORKER_EMBEDDER: "ArcFaceFrameEmbedder | None" = None
_CROSS_REF_WORKER_MATRIX: "np.ndarray | None" = None


def _init_cross_reference_worker(reference_matrix: np.ndarray) -> None:
    global _CROSS_REF_WORKER_EMBEDDER, _CROSS_REF_WORKER_MATRIX
    _CROSS_REF_WORKER_EMBEDDER = ArcFaceFrameEmbedder()
    _CROSS_REF_WORKER_MATRIX = reference_matrix


def _cross_reference_scores(
    video_dir: str, reference_names: list[str], reference_matrix: np.ndarray,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict[str, list[float]]:
    """Per-clip face-conditioned ID-similarity of every clip in ``video_dir`` against *each* of the 5
    reference identities (not just the one identity's clips this video_dir belongs to) -- one 30-long
    list per reference, in ``list_video_files`` order.

    This is gate criterion (c) (``docs/face_identity.md`` §5): calibrating ``IDENTITY_THRESHOLD``
    needs a real distribution to set a percentile-based threshold against (mirrors
    ``docs/imagenet_objects.md`` §5's methodology), not a single pooled scalar per identity pair --
    ``VideoFaceStats.detected`` is a *per-clip* decision (``id_sim_mean >= identity_threshold``), so
    the calibration's unit of observation has to be the clip, giving 30 same-identity and 120
    different-identity samples per identity here, not one aggregate number each.

    One embedding pass per clip suffices (``embed_frames`` is identity-agnostic); scoring against all
    five references is then one ``(num_faces, 512) @ (512, 5)`` matmul instead of five redundant
    detection passes. Degenerate frames (``zml/benchmarks/frame_quality.py``) are excluded the same
    way as everywhere else in this module -- a generation failure must not enter the calibration's
    negative distribution as if it were a real "definitely not this identity" measurement. A clip
    with no valid face-bearing frames (all no-face or all degenerate) contributes ``0.0`` to every
    reference, same convention as ``VideoFaceStats.id_sim_mean``.

    Parallel across clips (one process per video, mirrors ``VideoFaceDetector._score_videos_parallel``)
    -- this is the expensive half of ``score_existing(compute_cross_reference=True)``, an unavoidable
    second full embedding pass per clip on top of ``process_videos()``'s own, so it is not left
    serial the way the much smaller ``_clip_degenerate_flags`` pass is.
    """
    video_files = list_video_files(video_dir)
    paths = [os.path.join(video_dir, name) for name in video_files]
    if num_workers > 1 and len(paths) > 1:
        workers = min(num_workers, len(paths))
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_cross_reference_worker, initargs=(reference_matrix,)
        ) as pool:
            rows = list(pool.map(_cross_reference_one_clip, paths, chunksize=1))
    else:
        _init_cross_reference_worker(reference_matrix)
        rows = [_cross_reference_one_clip(path) for path in paths]

    per_clip: dict[str, list[float]] = {name: [] for name in reference_names}
    for row in rows:
        for ref, val in zip(reference_names, row.tolist()):
            per_clip[ref].append(val)
    return per_clip


def score_existing(
    output_dir: str,
    identity_prompts: dict[str, list[tuple[str, int]]],
    erased_identity: str | None = None,
    lora_checkpoint_dir: str | None = None,
    negative_prompt: str | None = None,
    identity_threshold: float = IDENTITY_THRESHOLD,
    skip_quality: bool = False,
    compute_cross_reference: bool = False,
) -> dict:
    """Score the videos already under ``output_dir`` and write ``id_similarity.json``.

    Split out of ``main`` at the pipeline boundary so re-scoring a finished run — after a metric or
    threshold change, say — costs minutes on a laptop (YuNet + ArcFace are ONNX-CPU) instead of
    regenerating 150 clips on a cluster.

    ``compute_cross_reference`` adds a second full embedding pass per identity (5x5 matrix, gate
    criterion (c) / ``IDENTITY_THRESHOLD`` calibration, ``docs/face_identity.md`` §5) and is opt-in:
    every other caller (NegPrompt baselines, reported erasure checkpoints) only needs the diagonal
    it already gets from ``process_videos()``, so paying for the off-diagonal on every eval would be
    pure waste for them.
    """
    eval_root = os.path.join(output_dir, f"eval_step_{EVAL_STEP}")
    # One embedder shared across all five identities; loading ArcFace/YuNet per identity is wasteful.
    embedder = ArcFaceFrameEmbedder()
    per_identity: dict[str, IdentityScores] = {}
    quality: dict[str, dict[str, float]] = {}
    cross_reference_per_clip: dict[str, dict[str, list[float]]] | None = None
    if compute_cross_reference:
        reference_names, reference_matrix = load_all_reference_embeddings()
        cross_reference_per_clip = {}

    for name, prompts in identity_prompts.items():
        video_dir = identity_video_dir(eval_root, name)
        detector = VideoFaceDetector(
            video_dir=video_dir, target_identity=name, identity_threshold=identity_threshold,
            embedder=embedder,
        )
        scores = detector.process_videos()
        collapse = _collapse_score(detector, video_dir)
        per_identity[name] = IdentityScores(
            id_sim=scores["face_id_similarity_mean"],
            id_sim_zerofill=scores["face_id_similarity_mean_zerofill"],
            face_present_rate=scores["face_present_rate"],
            identified_rate=scores["face_detection_rate"],
            num_videos=int(scores["total_videos"]),
            clips_without_face=int(scores["clips_without_face"]),
            clips_degenerate=int(scores["clips_degenerate"]),
            degenerate_frame_rate=scores["degenerate_frame_rate"],
            collapse_score=collapse,
        )
        if not skip_quality:
            degenerate_flags = _clip_degenerate_flags(video_dir)
            quality[name] = _quality_scores(video_dir, [p for p, _ in prompts], degenerate_flags)
        if compute_cross_reference:
            cross_reference_per_clip[name] = _cross_reference_scores(video_dir, reference_names, reference_matrix)
        s = per_identity[name]
        print(
            f"{name}: id_sim={s.id_sim:.4f} (zerofill {s.id_sim_zerofill:.4f}) "
            f"face_present_rate={s.face_present_rate:.4f} "
            f"clips_degenerate={s.clips_degenerate}/{s.num_videos} "
            f"collapse={'n/a' if collapse is None else f'{collapse:.4f}'}"
        )

    zerofilled = {name: s.zerofill() for name, s in per_identity.items()}
    # The mean 5x5 matrix (for a quick read) and the full per-clip data it was built from (for a
    # real percentile-based IDENTITY_THRESHOLD calibration, docs/face_identity.md §5) -- both derived
    # from cross_reference_per_clip, never recomputed, so they cannot disagree with each other.
    cross_reference_mean = None
    if cross_reference_per_clip is not None:
        cross_reference_mean = {
            clip_identity: {ref: float(np.mean(values)) for ref, values in refs.items()}
            for clip_identity, refs in cross_reference_per_clip.items()
        }
    report: dict = {
        "erased_identity": erased_identity,
        "lora_checkpoint_dir": lora_checkpoint_dir,
        "negative_prompt": negative_prompt,
        "identity_threshold": identity_threshold,
        "embedder": {
            "rec_sha256": embedder.rec_sha256,
            "det_sha256": embedder.det_sha256,
            "det_threshold": embedder.det_threshold,
            "min_face_px": embedder.min_face_px,
        },
        **_scores_block(per_identity, erased_identity),
        ZEROFILL_KEY: _scores_block(zerofilled, erased_identity),
        "quality": quality,
        "cross_reference": cross_reference_mean,
        "cross_reference_per_clip": cross_reference_per_clip,
    }
    with open(os.path.join(output_dir, "id_similarity.json"), "w") as f:
        json.dump(report, f, indent=2)

    headline = _headline(report, erased_identity)
    print(f"ID-Similarity (face-conditioned): {json.dumps(headline, indent=2)}")
    print(f"ID-Similarity (zero-filled):       {json.dumps(_headline(report[ZEROFILL_KEY], erased_identity), indent=2)}")
    return report


def _log_headlines(report: dict, erased_identity: str | None, disable_mlflow: bool) -> None:
    metrics = {f"eval/{k}": round(v, 4) for k, v in _headline(report, erased_identity).items()}
    metrics |= {
        f"eval/{ZEROFILL_KEY}/{k}": round(v, 4)
        for k, v in _headline(report[ZEROFILL_KEY], erased_identity).items()
    }
    if not disable_mlflow:
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=EVAL_STEP)
    wandb.log(metrics, step=EVAL_STEP)


def main(config: Config) -> dict:
    identity_prompts = load_identity_prompts(config.prompts_csv, config.eval_num_prompts_per_identity)
    eval_root = os.path.join(config.output_dir, f"eval_step_{EVAL_STEP}")
    os.makedirs(eval_root, exist_ok=True)

    pipe = build_eval_pipeline(config.model_id, config.lora_checkpoint_dir)
    pipe.transformer.eval()
    for name, prompts in identity_prompts.items():
        _generate_identity_videos(pipe, config, name, prompts, eval_root)

    report = score_existing(
        output_dir=config.output_dir,
        identity_prompts=identity_prompts,
        erased_identity=config.erased_identity,
        lora_checkpoint_dir=config.lora_checkpoint_dir,
        negative_prompt=config.resolved_negative_prompt(),
        identity_threshold=config.identity_threshold,
    )
    _log_headlines(report, config.erased_identity, config.disable_mlflow)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-score a finished face eval run in place.")
    parser.add_argument("--rescore", required=True, metavar="OUTPUT_DIR",
                        help="An outputs_TIMESTAMP dir holding eval_step_0/<identity_slug>/*.mp4")
    parser.add_argument("--prompts-csv", required=True, help="The prompt CSV the run was generated from")
    parser.add_argument("--erased-identity", default=None,
                        help="Omit for a base-model run (reports Erase/Preserve for every identity in turn)")
    parser.add_argument("--identity-threshold", type=float, default=IDENTITY_THRESHOLD)
    parser.add_argument("--skip-quality", action="store_true",
                        help="skip the CLIP/colorfulness/motion pass (needs torch); useful for a fast "
                             "threshold-only recalibration")
    parser.add_argument("--cross-reference", action="store_true",
                        help="also score every identity's clips against all 5 references (gate "
                             "criterion (c) / IDENTITY_THRESHOLD calibration, docs/face_identity.md "
                             "§5); one extra embedding pass per identity, opt-in since ordinary "
                             "reruns only need the diagonal")
    args = parser.parse_args()

    score_existing(
        output_dir=args.rescore,
        identity_prompts=load_identity_prompts(args.prompts_csv),
        erased_identity=args.erased_identity,
        identity_threshold=args.identity_threshold,
        skip_quality=args.skip_quality,
        compute_cross_reference=args.cross_reference,
    )
