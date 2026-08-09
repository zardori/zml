import json
import os
from dataclasses import dataclass
from typing import Callable, Protocol

import mlflow
import numpy as np
import pandas as pd
import wandb
from diffusers.utils import export_to_video
import torch

from zml.benchmarks.registry import build_detector
from zml.eval.clip_score import VideoClipScorer
from zml.eval.colorfulness import VideoColorfulnessScorer
from zml.eval.motion import VideoMotionScorer
from zml.eval.dover_scorer import DOVER_AVAILABLE, VideoDoverScorer


@dataclass
class EvalPrompt:
    prompt: str
    seed: int


def load_eval_prompts(path: str | None) -> list[EvalPrompt]:
    """Load a control-prompt CSV into ``EvalPrompt``s using the per-prompt seed baked into the file.

    This is the single canonical eval-prompt loader; every method should use it so results are
    comparable. Per the project seed policy (CLAUDE.md), evaluation seeds live in the CSV so that
    every experiment scores identical ``(prompt, seed)`` pairs.

    A seedless ``.txt`` prompt list is rejected rather than silently seeded. The ESD family used to
    fall back to index seeds (``EvalPrompt(p, 42 + i)``), which made those runs generate *different
    videos* from the CSV-based runs for the same prompts — exp058 (ESD) could not be compared to
    exp057 (frame_replace) because of it. Failing loudly keeps that from recurring.
    """
    if path is None:
        return []
    if not path.endswith(".csv"):
        raise ValueError(
            f"Eval prompts must come from a seeded CSV, got {path!r}. A .txt list has no seeds, so "
            f"loading it would score this run on different (prompt, seed) pairs than every CSV-based "
            f"experiment. Use the .csv equivalent (e.g. {path.rsplit('.', 1)[0]}.csv)."
        )
    df = pd.read_csv(path)
    missing = {"prompt", "seed"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s) {sorted(missing)}.")
    return [EvalPrompt(prompt=str(p), seed=int(s)) for p, s in zip(df["prompt"], df["seed"])]


# Detector-specific scores that only some concepts emit, logged to wandb/mlflow when present so the
# paper's comparison row can be read straight off a run. `nudity_frame_rate` is T2VUnlearning's
# "Nudity Rate" (see docs/comparability_t2vunlearning.md); it is deliberately kept alongside our own
# video-level `nudity_detection_rate` rather than replacing it, so historical runs stay readable.
OPTIONAL_EVAL_METRICS = ("nudity_frame_rate",)


def _round_metrics(obj: object, ndigits: int = 2) -> object:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [_round_metrics(v, ndigits) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_metrics(v, ndigits) for k, v in obj.items()}
    return obj


class EvalConfig(Protocol):
    output_dir: str
    eval_num_prompts: int
    eval_inference_steps: int


class EvalConceptConfig(Protocol):
    """The concept fields ``evaluate`` reads off whatever config it is handed."""

    concept: str
    concept_target: str | None


def score_video_dir(
    video_dir: str, prompts: list[str], concept: str, concept_target: str | None = None
) -> dict:
    """Run every metric over one directory of already-generated clips.

    Split out of ``evaluate`` so scoring can also be redone after the fact, without regenerating:
    an eval job that dies in this (CPU) phase still leaves a complete set of videos on disk, and
    re-running an hours-long generation to recover them would be pure waste. See
    ``tools/score_eval_videos.py``.

    ``prompts`` must be in the same order the clips were written (``video_0.mp4``, ``video_1.mp4``,
    ...), because CLIP score pairs the i-th clip with the i-th prompt; every scorer sorts its file
    listing, so that ordering holds as long as the caller passes the prompt list it generated from.
    """
    concept_scores = build_detector(concept, video_dir, concept_target).process_videos()
    clip_scores = VideoClipScorer(video_dir=video_dir, prompts=prompts).process_videos()
    colorfulness_scores = VideoColorfulnessScorer(video_dir=video_dir).process_videos()
    motion_scores = VideoMotionScorer(video_dir=video_dir).process_videos()
    dover_scores = (
        VideoDoverScorer(video_dir=video_dir).process_videos()
        if DOVER_AVAILABLE
        else {"technical": [], "aesthetic": []}
    )

    clip_arr = np.array(clip_scores) if clip_scores else np.array([0.0])
    color_arr = np.array(colorfulness_scores) if colorfulness_scores else np.array([0.0])
    motion_arr = np.array(motion_scores) if motion_scores else np.array([0.0])
    tech_arr = np.array(dover_scores["technical"]) if dover_scores["technical"] else np.array([0.0])
    aes_arr = np.array(dover_scores["aesthetic"]) if dover_scores["aesthetic"] else np.array([0.0])

    return {
        **concept_scores,
        # Concept-agnostic aliases so downstream (recorder/summary) reads one key across concepts.
        "concept_detection_rate": concept_scores[f"{concept}_detection_rate"],
        "concept_area_score_mean": concept_scores[f"{concept}_area_score_mean"],
        "clip_scores": clip_scores,
        "clip_score_mean": float(clip_arr.mean()),
        "clip_score_std": float(clip_arr.std()),
        "colorfulness_scores": colorfulness_scores,
        "colorfulness_mean": float(color_arr.mean()),
        "colorfulness_std": float(color_arr.std()),
        "motion_scores": motion_scores,
        "motion_score_mean": float(motion_arr.mean()),
        "motion_score_std": float(motion_arr.std()),
        "dover_technical_scores": dover_scores["technical"],
        "dover_technical_mean": float(tech_arr.mean()),
        "dover_technical_std": float(tech_arr.std()),
        "dover_aesthetic_scores": dover_scores["aesthetic"],
        "dover_aesthetic_mean": float(aes_arr.mean()),
        "dover_aesthetic_std": float(aes_arr.std()),
    }


def evaluate(
    pipe,
    transformer,
    config: EvalConfig,
    step: int,
    concept_prompts: list[EvalPrompt],
    related_prompts: list[EvalPrompt],
    unrelated_prompts: list[EvalPrompt],
    anchor_prompts: list[EvalPrompt] | None = None,
    prepare_for_prompt: Callable[[str], None] | None = None,
    log_mlflow: bool = True,
    include_related: bool = False,
) -> dict[str, dict]:
    was_training = transformer.training
    transformer.eval()
    eval_root = os.path.join(config.output_dir, f"eval_step_{step}")

    # Read off the config (rather than taken as a parameter) so training configs, which have no such
    # field, keep passing None and generate exactly as before. Applied to EVERY prompt set, not just
    # `concept`: NegPrompt is a deployed inference-time defense, so the collateral damage it does to
    # unrelated prompts is part of what is being measured.
    negative_prompt = getattr(config, "negative_prompt", None)

    # Every set other than `concept` is optional, and an absent one is *skipped* rather than
    # scored as an empty directory — a zero-filled row reads exactly like a real measurement of 0,
    # which is the mistake DOVER's 0.0-on-aarch64 already taught us once (see CLAUDE.md).
    # `related` additionally needs an explicit opt-in: it is wanted for standalone full-set eval but
    # skipped during training to save generation time.
    n = config.eval_num_prompts
    prompt_sets = {}
    if concept_prompts:
        prompt_sets["concept"] = concept_prompts[:n]
    if include_related and related_prompts:
        prompt_sets["related"] = related_prompts[:n]
    if unrelated_prompts:
        prompt_sets["unrelated"] = unrelated_prompts[:n]
    if anchor_prompts:
        prompt_sets["anchor"] = anchor_prompts[:n]
    if not prompt_sets:
        raise ValueError(
            "evaluate() was given no prompts in any set, so it would write a metrics.json of "
            "zeros — which is indistinguishable from a real measurement. Check the config's "
            "control_*_prompts fields."
        )

    with torch.no_grad():
        for set_name, eval_prompts in prompt_sets.items():
            video_dir = os.path.join(eval_root, set_name)
            os.makedirs(video_dir, exist_ok=True)
            for i, ep in enumerate(eval_prompts):
                if prepare_for_prompt is not None:
                    prepare_for_prompt(ep.prompt)
                result = pipe(
                    prompt=ep.prompt,
                    negative_prompt=negative_prompt,
                    num_frames=49,
                    num_inference_steps=config.eval_inference_steps,
                    generator=torch.Generator(device=pipe.device).manual_seed(ep.seed),
                )
                video_path = os.path.join(video_dir, f"video_{i}.mp4")
                export_to_video(result.frames[0], video_path, fps=8)
                print(f"Saved eval video: {video_path}")

    # Which concept the detector scores; defaults to fire so existing configs are unchanged. The
    # per-concept detector returns "<concept>_detection_rate"/"<concept>_area_score_mean" keys.
    # `concept_target` names the specific thing within a concept family (the ImageNet class for
    # concept "object"); it is unused by fire and nudity.
    concept = getattr(config, "concept", "fire")
    concept_target = getattr(config, "concept_target", None)
    rate_key, area_key = f"{concept}_detection_rate", f"{concept}_area_score_mean"

    metrics = {}
    for set_name, eval_prompts in prompt_sets.items():
        metrics[set_name] = score_video_dir(
            os.path.join(eval_root, set_name),
            [ep.prompt for ep in eval_prompts],
            concept,
            concept_target,
        )

    rounded_metrics = _round_metrics(metrics)
    metrics_path = os.path.join(eval_root, "metrics.json")
    # Written only when set, so every prior run's file keeps its exact schema, and under a key that
    # is not a prompt-set name; the returned dict stays clean for callers iterating prompt sets.
    on_disk = dict(rounded_metrics)
    if negative_prompt is not None:
        on_disk["_negative_prompt"] = negative_prompt
    with open(metrics_path, "w") as f:
        json.dump(on_disk, f, indent=2)
    print(f"Eval step {step}: {rounded_metrics}")

    if log_mlflow:
        for set_name, scores in metrics.items():
            mlflow.log_metric(f"eval/{set_name}_{rate_key}", round(scores[rate_key], 2), step=step)
            mlflow.log_metric(f"eval/{set_name}_{area_key}", round(scores[area_key], 4), step=step)
            mlflow.log_metric(f"eval/{set_name}_clip_score_mean", round(scores["clip_score_mean"], 2), step=step)
            mlflow.log_metric(f"eval/{set_name}_colorfulness_mean", round(scores["colorfulness_mean"], 2), step=step)
            mlflow.log_metric(f"eval/{set_name}_motion_score_mean", round(scores["motion_score_mean"], 3), step=step)
            if DOVER_AVAILABLE:
                mlflow.log_metric(f"eval/{set_name}_dover_technical_mean", round(scores["dover_technical_mean"], 2), step=step)
                mlflow.log_metric(f"eval/{set_name}_dover_aesthetic_mean", round(scores["dover_aesthetic_mean"], 2), step=step)
            for key in OPTIONAL_EVAL_METRICS:
                if key in scores:
                    mlflow.log_metric(f"eval/{set_name}_{key}", round(scores[key], 4), step=step)

    wandb_metrics = {
        f"eval/{set_name}_{k}": round(v, 4)
        for set_name, scores in metrics.items()
        for k, v in [
            (rate_key, scores[rate_key]),
            (area_key, scores[area_key]),
            ("clip_score_mean", scores["clip_score_mean"]),
            ("colorfulness_mean", scores["colorfulness_mean"]),
            ("motion_score_mean", scores["motion_score_mean"]),
        ]
    }
    if DOVER_AVAILABLE:
        wandb_metrics.update({
            f"eval/{set_name}_{k}": round(v, 2)
            for set_name, scores in metrics.items()
            for k, v in [
                ("dover_technical_mean", scores["dover_technical_mean"]),
                ("dover_aesthetic_mean", scores["dover_aesthetic_mean"]),
            ]
        })
    wandb_metrics.update({
        f"eval/{set_name}_{k}": round(scores[k], 4)
        for set_name, scores in metrics.items()
        for k in OPTIONAL_EVAL_METRICS
        if k in scores
    })
    wandb.log(wandb_metrics, step=step)

    if was_training:
        transformer.train()

    return metrics
