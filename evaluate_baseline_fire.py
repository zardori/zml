import os
import sys
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm.auto import tqdm
import gc
from diffusers.utils import export_to_video
import pandas as pd
import random
import json
from dataclasses import dataclass

@dataclass
class Config:
    model_id: str
    prompts_path: str
    control_related_prompts: str
    control_unrelated_prompts: str
    output_dir: str
    eval_num_prompts: int
    eval_inference_steps: int


def evaluate(pipe, transformer, config, step, concept_prompts, related_prompts, unrelated_prompts):
    sys.path.insert(0, os.path.dirname(__file__))
    from benchmarks.check_for_fire import VideoFireDetector

    transformer.eval()
    eval_root = os.path.join(config.output_dir, f"eval_step_{step}")

    prompt_sets = {
        "concept": random.sample(concept_prompts, min(config.eval_num_prompts, len(concept_prompts))),
        "related": random.sample(related_prompts, min(config.eval_num_prompts, len(related_prompts))),
        "unrelated": random.sample(unrelated_prompts, min(config.eval_num_prompts, len(unrelated_prompts))),
    }

    with torch.no_grad():
        for set_name, prompts in prompt_sets.items():
            video_dir = os.path.join(eval_root, set_name)
            os.makedirs(video_dir, exist_ok=True)
            for i, prompt in enumerate(prompts):
                result = pipe(
                    prompt=prompt,
                    num_frames=49,
                    num_inference_steps=config.eval_inference_steps,
                    generator=torch.Generator(device=pipe.device).manual_seed(42 + i),
                )
                video_path = os.path.join(video_dir, f"video_{i}.mp4")
                export_to_video(result.frames[0], video_path, fps=8)
                print(f"Saved eval video: {video_path}")

    metrics = {}
    for set_name in prompt_sets:
        video_dir = os.path.join(eval_root, set_name)
        detector = VideoFireDetector(video_dir=video_dir)
        scores = detector.process_videos()
        metrics[set_name] = scores

    metrics_path = os.path.join(eval_root, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Eval step {step}: {metrics}")

    transformer.train()
    transformer.requires_grad_(False)


def main(config: Config):
    data = pd.read_csv(config.prompts_path)
    CONCEPT_PROMPTS = data["prompt"].tolist()

    with open(config.control_related_prompts) as f:
        RELATED_PROMPTS = [l.strip() for l in f if l.strip()]
    with open(config.control_unrelated_prompts) as f:
        UNRELATED_PROMPTS = [l.strip() for l in f if l.strip()]

    # @title 2. Setup Configuration
    # Using CogVideoX-2b (Fits on approx 12-16GB VRAM with optimizations)
    MODEL_ID = config.model_id
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 # CogVideoX works best with bf16

    # Load Pipeline
    pipe = CogVideoXPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE
    ).to(DEVICE)

    # Enable memory savings
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 1. Extract the Transformer (The noise predictor)
    transformer = pipe.transformer
    transformer.train()
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing() # Critical for VRAM

    evaluate(pipe, transformer, config, 0,
                CONCEPT_PROMPTS, RELATED_PROMPTS, UNRELATED_PROMPTS)

    print("Training Complete.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default="THUDM/CogVideoX-5b")
    parser.add_argument("--prompts_path", type=str, default="prompts/cogvideox_fire.csv")
    parser.add_argument("--control_related_prompts", type=str, default="prompts/cogvideox_fire_control_related.txt")
    parser.add_argument("--control_unrelated_prompts", type=str, default="prompts/cogvideox_fire_control_unrelated.txt")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--eval_num_prompts", type=int, default=3)
    parser.add_argument("--eval_inference_steps", type=int, default=20)
    args = parser.parse_args()
    config = Config(**vars(args))
    main(config)
