import os
import subprocess
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


data = pd.read_csv('prompts/cogvideox_nudity.csv')

CONCEPT_PROMPTS = data["prompt"].tolist()

# @title 2. Setup Configuration
# Using CogVideoX-2b (Fits on approx 12-16GB VRAM with optimizations)
MODEL_ID = "THUDM/CogVideoX-5b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 # CogVideoX works best with bf16

# Guidance scale for erasure (How hard we push AWAY)
NEGATIVE_GUIDANCE_SCALE = 2.0
STEPS = 1000 # Fewer steps needed for DiT usually


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
    r=8, # Slightly higher rank for video
    lora_alpha=8,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.0,
    bias="none",
)

# 3. Inject LoRA
transformer = get_peft_model(transformer, lora_config)
transformer.print_trainable_parameters()

optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-3)

# Helper to encode text specific to CogVideoX
def encode_prompt(pipe, prompt):
    # CogVideoX uses T5 and requires specific embedding handling
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=False # We handle CFG manually in loop
    )
    return prompt_embeds.to(DEVICE, dtype=DTYPE)

# Pre-calculate embeddings to save VRAM
# print("Pre-computing embeddings...")

# Unload the text encoder to free up VRAM for training
# del pipe.text_encoder
# gc.collect()
# torch.cuda.empty_cache()

print("Starting ESD Training for Video...")

scheduler = pipe.scheduler

# Latent setup for CogVideoX
# Initial Generation Shape: [Batch, Channels, Frames, Height, Width]
batch_size = 1
num_channels = 16
num_frames = 13
height = 60
width = 90

pbar = tqdm(range(STEPS))

for step in pbar:
    CONCEPT_PROMPT = random.choice(CONCEPT_PROMPTS)
    # print(CONCEPT_PROMPTS)
    # print(CONCEPT_PROMPT)
    # print(f"Generating Video for: {CONCEPT_PROMPT}")
    with torch.no_grad():
        concept_emb = encode_prompt(pipe, CONCEPT_PROMPT)
        null_emb = encode_prompt(pipe, "")


    optimizer.zero_grad()

    # 1. Prepare 3D Latents (x_t)
    # We keep this in (B, C, F, H, W) for the scheduler
    latents = torch.randn(
        (batch_size, num_channels, num_frames, height, width),
        device=DEVICE,
        dtype=DTYPE
    )

    # 2. Timesteps
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (batch_size,),
        device=DEVICE
    ).long()

    # 3. Add Noise
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    # --- FIX: PERMUTE DIMENSIONS FOR TRANSFORMER ---
    # Convert [B, C, F, H, W] -> [B, F, C, H, W]
    # The transformer expects Frames at index 1, Channels at index 2
    model_input = noisy_latents.permute(0, 2, 1, 3, 4)
    # -----------------------------------------------

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
            target = model_pred_uncond - NEGATIVE_GUIDANCE_SCALE * (model_pred_text - model_pred_uncond)

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
        lora_output_dir = f"./cogvideox_erasure_lora_nudity_step{step + 1}"
        os.makedirs(lora_output_dir, exist_ok=True)
        transformer.save_pretrained(lora_output_dir)
        print(f"Checkpoint saved to: {lora_output_dir}")
        subprocess.run(
            ["zip", "-r", f'{lora_output_dir}.zip', lora_output_dir],
            check=True
        )

print("Training Complete.")