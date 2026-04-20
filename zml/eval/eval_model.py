import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from peft import PeftModel

from zml.eval.check_for_fire import VideoFireDetector
from zml.eval.clip_score import VideoClipScorer
from zml.eval.dover_scorer import VideoDoverScorer


@dataclass
class Config:
    model_id: str
    control_concept_prompts: str  # CSV: prompt,seed columns
    control_related_prompts: str  # CSV: prompt,seed columns
    control_unrelated_prompts: str  # CSV: prompt,seed columns
    output_dir: str
    eval_num_prompts: int
    eval_inference_steps: int
    lora_checkpoint_dir: str | None = None


def _load_prompts_csv(path: str) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path)
    return df["prompt"].tolist(), df["seed"].tolist()


def _generate_videos(
    pipe: CogVideoXPipeline,
    prompts: list[str],
    seeds: list[int],
    video_dir: str,
    num_inference_steps: int,
) -> None:
    os.makedirs(video_dir, exist_ok=True)
    with torch.no_grad():
        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            result = pipe(
                prompt=prompt,
                num_frames=49,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=pipe.device).manual_seed(seed),
            )
            video_path = os.path.join(video_dir, f"video_{i}.mp4")
            export_to_video(result.frames[0], video_path, fps=8)
            print(f"Saved eval video: {video_path}")


def _score_videos(video_dir: str, prompts: list[str]) -> dict:
    fire_scores = VideoFireDetector(video_dir=video_dir).process_videos()
    clip_scores = VideoClipScorer(video_dir=video_dir, prompts=prompts).process_videos()
    dover_scores = VideoDoverScorer(video_dir=video_dir).process_videos()

    clip_arr = np.array(clip_scores) if clip_scores else np.array([0.0])
    tech_arr = np.array(dover_scores["technical"]) if dover_scores["technical"] else np.array([0.0])
    aes_arr = np.array(dover_scores["aesthetic"]) if dover_scores["aesthetic"] else np.array([0.0])

    return {
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


def main(config: Config) -> dict:
    concept_prompts, concept_seeds = _load_prompts_csv(config.control_concept_prompts)
    related_prompts, related_seeds = _load_prompts_csv(config.control_related_prompts)
    unrelated_prompts, unrelated_seeds = _load_prompts_csv(config.control_unrelated_prompts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = CogVideoXPipeline.from_pretrained(config.model_id, torch_dtype=torch.bfloat16).to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if config.lora_checkpoint_dir is not None:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, config.lora_checkpoint_dir)
        print(f"Loaded LoRA checkpoint from {config.lora_checkpoint_dir}")

    pipe.transformer.eval()

    n = config.eval_num_prompts
    prompt_sets = {
        "concept": (concept_prompts[:n], concept_seeds[:n]),
        "related": (related_prompts[:n], related_seeds[:n]),
        "unrelated": (unrelated_prompts[:n], unrelated_seeds[:n]),
    }

    os.makedirs(config.output_dir, exist_ok=True)
    metrics = {}
    for set_name, (prompts, seeds) in prompt_sets.items():
        video_dir = os.path.join(config.output_dir, set_name)
        _generate_videos(pipe, prompts, seeds, video_dir, config.eval_inference_steps)
        metrics[set_name] = _score_videos(video_dir, prompts)

    metrics_path = os.path.join(config.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation complete. Metrics saved to {metrics_path}")
    print(metrics)

    return metrics
