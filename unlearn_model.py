import os
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
    prompts_path: str
    lora_rank: int
    lora_alpha: float
    negative_guidance_scale: float
    steps: int
    learning_rate: float
    lora_dropout: float
    output_dir: str


def main(config: Config):
    data = pd.read_csv(config.prompts_path)
    CONCEPT_PROMPTS = data["prompt"].tolist()

    # @title 2. Setup Configuration
    # Using CogVideoX-2b (Fits on approx 12-16GB VRAM with optimizations)
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

    print("Starting ESD Training for Video...")

    scheduler = pipe.scheduler

    # Latent setup for CogVideoX
    # Initial Generation Shape: [Batch, Channels, Frames, Height, Width]
    batch_size = 1
    num_channels = 16
    num_frames = 13
    height = 60
    width = 90

    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps)

    pbar = tqdm(range(config.steps))

    for step in pbar:
        CONCEPT_PROMPT = random.choice(CONCEPT_PROMPTS)
        with torch.no_grad():
            concept_emb = encode_prompt(pipe, CONCEPT_PROMPT)
            null_emb = encode_prompt(pipe, "")

        optimizer.zero_grad()

        # 1. Sample a random denoising step t (not the very first step)
        t_idx = random.randint(1, num_inference_steps - 1)
        t = scheduler.timesteps[t_idx]
        timesteps = t.unsqueeze(0).expand(batch_size).to(DEVICE)

        # 2. Start from pure noise x_T [B, C, F, H, W]
        latents = torch.randn(
            (batch_size, num_channels, num_frames, height, width),
            device=DEVICE,
            dtype=DTYPE
        )

        # 3. Partially denoise with the student model from T down to t
        with torch.no_grad():
            for ts in scheduler.timesteps:
                if ts <= t:
                    break
                model_input = latents.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W] -> [B, F, C, H, W]
                noise_pred = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=concept_emb,
                    timestep=ts.unsqueeze(0).expand(batch_size).to(DEVICE),
                ).sample.permute(0, 2, 1, 3, 4)  # back to [B, C, F, H, W]
                latents = scheduler.step(noise_pred, ts, latents).prev_sample

        # Convert to transformer format [B, F, C, H, W] for ESD loss computation
        model_input = latents.permute(0, 2, 1, 3, 4)

        # -----------------------------------------------------------
        # TEACHER STEP (Frozen DiT)
        # -----------------------------------------------------------
        with torch.no_grad():
            with transformer.disable_adapter():
                # Unconditional
                model_pred_uncond = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=null_emb,
                    timestep=timesteps,
                ).sample

                # Conditional (Concept)
                model_pred_text = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=concept_emb,
                    timestep=timesteps,
                ).sample

                # ESD Target Calculation
                target = model_pred_uncond - config.negative_guidance_scale * (model_pred_text - model_pred_uncond)

        # -----------------------------------------------------------
        # STUDENT STEP (Trainable LoRA)
        # -----------------------------------------------------------
        model_pred = transformer(
            hidden_states=model_input,
            encoder_hidden_states=concept_emb,
            timestep=timesteps,
        ).sample

        # Loss
        loss = F.mse_loss(model_pred.float(), target.float())

        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.4f}")

        if (step + 1) % 200 == 0:
            lora_output_dir = os.path.join(config.output_dir, f"cogvideox_erasure_lora_nudity_step{step + 1}")
            os.makedirs(lora_output_dir, exist_ok=True)
            transformer.save_pretrained(lora_output_dir)
            print(f"Checkpoint saved to: {lora_output_dir}")

    print("Training Complete.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prompts_path", type=str, default="prompts/cogvideox_nudity.csv")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--negative_guidance_scale", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    config = Config(**vars(args))
    main(config)