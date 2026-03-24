import os
import subprocess
from argparse import ArgumentParser
import time
from datetime import datetime

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
from typing import Optional

def weighted_sample(traj, config: "Config"):
    steps_in_traj = torch.tensor([e["step"] for e in traj], dtype=torch.float)
    steps_normalized = steps_in_traj / steps_in_traj.max()  # scale to [0, 1]
    probs = torch.softmax(steps_normalized / config.sampling_temperature, dim=0)
    return random.choices(traj, weights=probs.tolist(), k=1)[0]

sampling_strategies = {
    "uniform": lambda traj, _: random.choice(traj),
    "weighted": weighted_sample,
}


@dataclass
class Config:
    metadata_file: str  # Entries containing (prompt, seed, step, name of a file with latent)
    metadata_count: Optional[int]
    latents_dir: str
    lora_rank: int
    lora_alpha: float
    negative_guidance_scale: float
    steps: int
    learning_rate: float
    lora_dropout: float
    output_dir: str
    step_sampling_strategy: str
    sampling_temperature: float


def main(config: Config):
    training_start = time.time()
    formatted_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Started training procedure at: {formatted_start}")
    MODEL_ID = "THUDM/CogVideoX-5b"
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

    # 2. Define LoRA Config for Transformer
    # Targeting the attention projections in the DiT
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=config.lora_dropout,
        bias="none",
    )

    # 3. Inject LoRA
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate)

    # Helper to encode text specific to CogVideoX
    def encode_prompt(pipe, prompt):
        # CogVideoX uses T5 and requires specific embedding handling
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False # We handle CFG manually in loop
        )
        return prompt_embeds.to(DEVICE, dtype=DTYPE)

    # Load metadata
    with open(config.metadata_file) as f:
        metadata = json.load(f)

    if config.metadata_count:
        metadata = metadata[:config.metadata_count]

    # Group by (prompt, seed) so we can sample full trajectories
    from collections import defaultdict
    trajectories = defaultdict(list)
    for entry in metadata:
        key = (entry["prompt"], entry["seed"])
        trajectories[key].append(entry)

    # Sort each trajectory by descending timestep (noisy → clean)
    for key in trajectories:
        trajectories[key].sort(key=lambda x: x["step"], reverse=True)
    traj_keys = list(trajectories.keys())

    # Cache prompt embeddings to save time and memory during training
    prompt_emb_cache = {}
    with torch.no_grad():
        for key in traj_keys:
            prompt = key[0]
            if prompt not in prompt_emb_cache:
                prompt_emb_cache[prompt] = encode_prompt(pipe, prompt)
        null_emb = encode_prompt(pipe, "")

    del pipe.text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    print("Starting ESD Training for Video...")
    pbar = tqdm(range(config.steps))
    for step in pbar:
        optimizer.zero_grad()

        # 1. Sample a random trajectory (prompt + seed)
        key = random.choice(traj_keys)
        traj = trajectories[key]
        prompt, seed = key

        entry = sampling_strategies[config.step_sampling_strategy](traj, config)
        target_step = torch.tensor([entry["step"]], dtype=torch.long, device=DEVICE)

        # 3. Load the saved latent at that timestep
        x_t = torch.load(
            os.path.join(config.latents_dir, entry["latent_path"]),
            map_location=DEVICE
        ).to(dtype=DTYPE)

        concept_emb = prompt_emb_cache[prompt]
        model_input = x_t.permute(0, 2, 1, 3, 4)  # [B,C,F,H,W] → [B,F,C,H,W]

        # Teacher (frozen base model)
        with torch.no_grad():
            with transformer.disable_adapter():
                model_pred_uncond = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=null_emb,
                    timestep=target_step,
                ).sample

                model_pred_text = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=concept_emb,
                    timestep=target_step,
                ).sample

        # ESD target
        target = model_pred_uncond - config.negative_guidance_scale * (model_pred_text - model_pred_uncond)

        # Student (LoRA active)
        model_pred = transformer(
            hidden_states=model_input,
            encoder_hidden_states=concept_emb,
            timestep=target_step,
        ).sample

        loss = F.mse_loss(model_pred.float(), target.float())
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item():.4f}")

        if (step + 1) % 200 == 0:
            lora_output_dir = os.path.join(config.output_dir, f"cogvideox_erasure_lora_nudity_step{step + 1}")
            os.makedirs(lora_output_dir, exist_ok=True)
            transformer.save_pretrained(lora_output_dir)
            print(f"Checkpoint saved to: {lora_output_dir}")
            subprocess.run(
                ["zip", "-r", f'{lora_output_dir}.zip', lora_output_dir],
                check=True
            )

    elapsed = time.time() - training_start
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training Complete. Finished at {finished_at}, took {int(hours)}h {int(minutes)}m {seconds:.1f}s.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--metadata_file", type=str, default="unlearning_dataset/metadata.json")
    parser.add_argument("--metadata_count", type=int, default=None)
    parser.add_argument("--latents_dir", type=str, default="unlearning_dataset/latents")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--negative_guidance_scale", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--step_sampling_strategy", type=str,
                        default="uniform", choices=sampling_strategies.keys(),
                        help="How to sample steps from the trajectory")
    # Temperature used when sampling is "weighted". Lower -> more focus on early generation steps
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    args = parser.parse_args()
    config = Config(**vars(args))
    if config.step_sampling_strategy == "weighted" and config.sampling_temperature <= 0:
        raise ValueError("sampling_temperature must be > 0 for weighted strategy")
    main(config)