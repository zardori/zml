import torch
import pandas as pd
import json
import os
from diffusers import CogVideoXPipeline
from tqdm import tqdm

# Configuration
CSV_PATH = "../../prompts/cogvideox_nudity.csv"  # Your file with 'prompt' and 'seed'
SAVE_DIR = "./unlearning_dataset"
LATENT_DIR = os.path.join(SAVE_DIR, "latents")
os.makedirs(LATENT_DIR, exist_ok=True)

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16).to("cuda")
scheduler = pipe.scheduler
scheduler.set_timesteps(50)

scheduler.alphas_cumprod = scheduler.alphas_cumprod.to("cuda")

metadata = []

with torch.no_grad():
    df = pd.read_csv(CSV_PATH)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['prompt']
        seed = int(row['seed'])
        
        # Encode prompt once
        prompt_embeds, _ = pipe.encode_prompt(prompt=prompt, do_classifier_free_guidance=False)
        prompt_embeds = prompt_embeds.to("cuda", dtype=torch.bfloat16)
        
        # Initial Latents
        generator = torch.Generator(device="cuda").manual_seed(seed)
        latents = torch.randn((1, 16, 13, 60, 90), device="cuda", dtype=torch.bfloat16, generator=generator)
        
        # Denoising Loop
        for i, t in enumerate(scheduler.timesteps):
            # 1. Save current state (The tuple: prompt, seed, step, latent)
            latent_filename = f"p{idx}_s{seed}_step{t.item()}.pt"
            torch.save(latents.cpu(), os.path.join(LATENT_DIR, latent_filename))
            metadata.append({
                "prompt": prompt,
                "seed": seed,
                "step": t.item(),
                "latent_path": latent_filename
            })
            # 2. Step forward to get the next latent in the trajectory
            model_input = latents.permute(0, 2, 1, 3, 4)
            noise_pred = pipe.transformer(
                hidden_states=model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=torch.tensor([t], device="cuda")
            ).sample.permute(0, 2, 1, 3, 4)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

# Save mapping for the trainer
with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f)

print(f"Dataset generated. Total entries: {len(metadata)}")