import os
from dataclasses import dataclass

import pandas as pd
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from peft import PeftModel


@dataclass
class GenerateConfig:
    model_id: str
    output_dir: str
    prompts_file: str  # path to a seeded .csv ('prompt','seed') or a plain .txt (one per line)
    num_inference_steps: int = 50
    num_frames: int = 49
    guidance_scale: float = 6.0
    fps: int = 8
    # Base seed for .txt prompts; ignored for .csv (which carries per-prompt seeds).
    # None means generate non-deterministically to inspect output diversity.
    seed: int | None = None
    lora_checkpoint_dir: str | None = None


@dataclass
class GenPrompt:
    prompt: str
    seed: int | None


def load_prompts(path: str, default_seed: int | None) -> list[GenPrompt]:
    """Load prompts from a seeded .csv (using its 'seed' column, per the eval seed policy)
    or a plain .txt with one prompt per line (each assigned ``default_seed``)."""
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        return [GenPrompt(prompt, int(seed)) for prompt, seed in zip(df["prompt"], df["seed"])]
    with open(path) as f:
        return [GenPrompt(line.strip(), default_seed) for line in f if line.strip()]


def _build_pipeline(config: GenerateConfig) -> CogVideoXPipeline:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = CogVideoXPipeline.from_pretrained(config.model_id, torch_dtype=torch.bfloat16).to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if config.lora_checkpoint_dir is not None:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, config.lora_checkpoint_dir)
        print(f"Loaded LoRA checkpoint from {config.lora_checkpoint_dir}")
    return pipe


def main(config: GenerateConfig) -> None:
    prompts = load_prompts(config.prompts_file, config.seed)
    pipe = _build_pipeline(config)
    os.makedirs(config.output_dir, exist_ok=True)

    with torch.no_grad():
        for i, gp in enumerate(prompts):
            generator = (
                torch.Generator(device=pipe.device).manual_seed(gp.seed)
                if gp.seed is not None
                else None
            )
            result = pipe(
                prompt=gp.prompt,
                num_frames=config.num_frames,
                guidance_scale=config.guidance_scale,
                num_inference_steps=config.num_inference_steps,
                generator=generator,
            )
            video_path = os.path.join(config.output_dir, f"{i:03d}.mp4")
            export_to_video(result.frames[0], video_path, fps=config.fps)
            print(f"Saved video: {video_path}")
