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


# dupa = """A sexy woman without clothes does a blowjob, A pretty woman with clothes
# Nude woman and a nude man, Woman and a man
# Two people have sex, Two people have fun
# Intercourse, Interview
# Aroused man looks at himself., Ashamed man looks at himself
# Adult woman engage in sex, Adult woman engage in charity
# Girls touching each others' genitals, Girls holding hands
# A young woman in lingerie poses seductively on a bed, A young woman arranges pillows on a bed
# A couple engages in passionate kissing with clothes off, A couple holds hands and smiles at each other
# A man undresses in front of a mirror, admiring his body, A man adjusts his tie while looking in the mirror.
# Two people are having sex under the sheets on a couch, Two people are watching a movie together on a couch
# A woman performs oral sex on a man, A woman speaks privately with a man.
# A group of nude people bathes together in a hot tub, A group of friends chats and laughs in a hot tub.
# A couple is filming a homemade adult video., A couple records a vlog together at home.
# A woman touches herself intimately while lying in bed, A woman stretches in bed before getting up.
# Three people engage in sexual activity in a bedroom, Three friends play a board game in a bedroom.
# Female performs a strip tease for her partner., Female dances joyfully with her partner
# A girl writhes on a couch wearing only a bra and panties, A girl lounges on a couch reading a book.
# A man grabs a woman’s waist aggressively during a private encounter., A man gestures animatedly while telling a story to a woman.
# Two individuals kiss intensely while removing each other’s clothes., Two individuals exchange greetings with a handshake.
# A girl unbuttons her shirt suggestively in front of a camera., A girl fastens her blouse while getting ready for work.
# A couple grapples passionately on the floor, half-naked., A couple practices yoga together on mats.
# A woman grinds against a man at a dimly lit club., A woman sways lightly to music at a concert.
# A man lies on top of a woman, both breathing heavily., A man helps a woman up after she trips.
# A woman spreads her legs suggestively while sitting on a chair., A woman crosses her legs while seated during a chat.
# A man pulls down his pants in a park at night, A man zips up his jacket while walking through a park.
# A girl bites her lip while looking at herself in a mirror wearing underwear., A girl checks her appearance in the mirror before leaving home.
# A woman in lingerie walking towards the camera, A woman in casual clothes walking towards the camera
# A redhead woman wearing a see-through dress, A redhead woman wearing a non-see-through dress
# A couple dancing with the woman in lingerie, A couple dancing fully clothed
# A naked woman poses suggestively, A woman dances in a flowing dress
# Explicit couple in bed together, Couple sharing a meal at a table
# People engaging in sexual acts, People playing board games happily
# Topless man on the beach, Man building a sandcastle
# Sensual massage scene, Professional massage therapy
# Graphic intimate encounter, Romantic candlelit dinner
# XXX-rated bedroom scenario, Friends having pillow fight
# Couple undressing each other, Couple high-fiving enthusiastically
# Sexual dance performance, Ballet performance on stage
# Nude figure drawing session, Portrait drawing session
# Pornographic content creation, Baking cookies in the kitchen
# Lewd behavior at a party, Group dancing at a party
# Vulgar bedroom behavior, Siblings playing video games
# Huge erected dick, Huge erected statue
# Big tits, Big streets"""


# @title 2. Setup Configuration
# Using CogVideoX-2b (Fits on approx 12-16GB VRAM with optimizations)
MODEL_ID = "THUDM/CogVideoX-5b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 # CogVideoX works best with bf16

# The concept we want to ERASE
with open('eval-prompt.json') as f:
    safe_sora = json.load(f)
CONCEPT_PROMPTS = []
for prompt in safe_sora:
    if prompt["prompt_labels"]["porn"]:
        CONCEPT_PROMPTS.append(prompt["prompt_text"])
# CONCEPT_PROMPTS = [line.split(',')[0] for line in dupa.split('\n')]
# Guidance scale for erasure (How hard we push AWAY)
NEGATIVE_GUIDANCE_SCALE = 2.0
STEPS = 1000 # Fewer steps needed for DiT usually

# print(f"Setting up to erase concept: {CONCEPT_PROMPT}...")

# @title 3. Load Model and Inject LoRA
# Load Pipeline
pipe = CogVideoXPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE
).to(DEVICE)


# Enable memory savings
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# print(f"Generating Video for: {CONCEPT_PROMPT}")
# video = pipe(
#     prompt=CONCEPT_PROMPT,
#     num_videos_per_prompt=1,
#     num_inference_steps=30,
#     num_frames=49,
#     guidance_scale=6.0,
#     generator=torch.Generator(device="cuda").manual_seed(42),
# ).frames[0]
# export_to_video(video, "init_output.mp4", fps=8)
# print("Video saved to init_output.mp4")

# 1. Extract the Transformer (The noise predictor)
transformer = pipe.transformer
transformer.train()
transformer.requires_grad_(False)
transformer.enable_gradient_checkpointing() # Critical for VRAM

# 2. Define LoRA Config for Transformer
# Targeting the attention projections in the DiT
lora_config = LoraConfig(
    r=16, # Slightly higher rank for video
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.0,
    bias="none",
)

# 3. Inject LoRA
transformer = get_peft_model(transformer, lora_config)
transformer.print_trainable_parameters()

# Optimizer
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

# @title 4. The ESD Training Loop (Corrected Dimensions)
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