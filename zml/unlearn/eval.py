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

from zml.eval.check_for_fire import VideoFireDetector
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


def _concept_detector(concept: str, video_dir: str):
    """Return the video detector for ``concept``. The nudity detector is imported lazily so the fire
    path (and every trainer that imports this module) never requires ``nudenet`` to be installed."""
    if concept == "fire":
        return VideoFireDetector(video_dir=video_dir)
    if concept == "nudity":
        from zml.benchmarks.check_for_nudity import VideoNudeDetector
        return VideoNudeDetector(video_dir=video_dir)
    raise ValueError(f"Unknown concept {concept!r}; expected 'fire' or 'nudity'.")


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

    # `related` is skipped during training (compute) but wanted for standalone full-set
    # eval; it is included only when explicitly requested and actually provided.
    n = config.eval_num_prompts
    prompt_sets = {"concept": concept_prompts[:n]}
    if include_related and related_prompts:
        prompt_sets["related"] = related_prompts[:n]
    prompt_sets["unrelated"] = unrelated_prompts[:n]
    if anchor_prompts:
        prompt_sets["anchor"] = anchor_prompts[:n]

    with torch.no_grad():
        for set_name, eval_prompts in prompt_sets.items():
            video_dir = os.path.join(eval_root, set_name)
            os.makedirs(video_dir, exist_ok=True)
            for i, ep in enumerate(eval_prompts):
                if prepare_for_prompt is not None:
                    prepare_for_prompt(ep.prompt)
                result = pipe(
                    prompt=ep.prompt,
                    num_frames=49,
                    num_inference_steps=config.eval_inference_steps,
                    generator=torch.Generator(device=pipe.device).manual_seed(ep.seed),
                )
                video_path = os.path.join(video_dir, f"video_{i}.mp4")
                export_to_video(result.frames[0], video_path, fps=8)
                print(f"Saved eval video: {video_path}")

    # Which concept the detector scores; defaults to fire so existing configs are unchanged. The
    # per-concept detector returns "<concept>_detection_rate"/"<concept>_area_score_mean" keys.
    concept = getattr(config, "concept", "fire")
    rate_key, area_key = f"{concept}_detection_rate", f"{concept}_area_score_mean"

    metrics = {}
    for set_name, eval_prompts in prompt_sets.items():
        video_dir = os.path.join(eval_root, set_name)
        concept_scores = _concept_detector(concept, video_dir).process_videos()
        clip_scores = VideoClipScorer(
            video_dir=video_dir, prompts=[ep.prompt for ep in eval_prompts]
        ).process_videos()
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

        metrics[set_name] = {
            **concept_scores,
            # Concept-agnostic aliases so downstream (recorder/summary) reads one key across concepts.
            "concept_detection_rate": concept_scores[rate_key],
            "concept_area_score_mean": concept_scores[area_key],
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

    rounded_metrics = _round_metrics(metrics)
    metrics_path = os.path.join(eval_root, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(rounded_metrics, f, indent=2)
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
    wandb.log(wandb_metrics, step=step)

    if was_training:
        transformer.train()

    return metrics
