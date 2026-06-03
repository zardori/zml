import json
import os
from dataclasses import dataclass
from typing import Callable, Protocol

import mlflow
import numpy as np
import wandb
from diffusers.utils import export_to_video
import torch

from zml.eval.check_for_fire import VideoFireDetector
from zml.eval.clip_score import VideoClipScorer
from zml.eval.dover_scorer import DOVER_AVAILABLE, VideoDoverScorer


@dataclass
class EvalPrompt:
    prompt: str
    seed: int


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
) -> dict[str, dict]:
    was_training = transformer.training
    transformer.eval()
    eval_root = os.path.join(config.output_dir, f"eval_step_{step}")

    prompt_sets = {
        "concept": concept_prompts[: config.eval_num_prompts],
        # "related": related_prompts[: config.eval_num_prompts],
        "unrelated": unrelated_prompts[: config.eval_num_prompts],
        "anchor": anchor_prompts[: config.eval_num_prompts],
    } if anchor_prompts else {
        "concept": concept_prompts[: config.eval_num_prompts],
        # "related": related_prompts[: config.eval_num_prompts],
        "unrelated": unrelated_prompts[: config.eval_num_prompts],
    }

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

    metrics = {}
    for set_name, eval_prompts in prompt_sets.items():
        video_dir = os.path.join(eval_root, set_name)
        fire_scores = VideoFireDetector(video_dir=video_dir).process_videos()
        clip_scores = VideoClipScorer(
            video_dir=video_dir, prompts=[ep.prompt for ep in eval_prompts]
        ).process_videos()
        dover_scores = (
            VideoDoverScorer(video_dir=video_dir).process_videos()
            if DOVER_AVAILABLE
            else {"technical": [], "aesthetic": []}
        )

        clip_arr = np.array(clip_scores) if clip_scores else np.array([0.0])
        tech_arr = np.array(dover_scores["technical"]) if dover_scores["technical"] else np.array([0.0])
        aes_arr = np.array(dover_scores["aesthetic"]) if dover_scores["aesthetic"] else np.array([0.0])

        metrics[set_name] = {
            **fire_scores,
            "clip_scores": clip_scores,
            "clip_score_mean": float(clip_arr.mean()),
            "clip_score_std": float(clip_arr.std()),
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

    for set_name, scores in metrics.items():
        mlflow.log_metric(f"eval/{set_name}_fire_detection_rate", round(scores["fire_detection_rate"], 2), step=step)
        mlflow.log_metric(f"eval/{set_name}_clip_score_mean", round(scores["clip_score_mean"], 2), step=step)
        if DOVER_AVAILABLE:
            mlflow.log_metric(f"eval/{set_name}_dover_technical_mean", round(scores["dover_technical_mean"], 2), step=step)
            mlflow.log_metric(f"eval/{set_name}_dover_aesthetic_mean", round(scores["dover_aesthetic_mean"], 2), step=step)

    wandb_metrics = {
        f"eval/{set_name}_{k}": round(v, 2)
        for set_name, scores in metrics.items()
        for k, v in [
            ("fire_detection_rate", scores["fire_detection_rate"]),
            ("clip_score_mean", scores["clip_score_mean"]),
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
    transformer.requires_grad_(False)
    return metrics
