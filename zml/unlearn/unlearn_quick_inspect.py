import os
import gc
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm.auto import tqdm


def encode_prompt(pipe, prompt, device, dtype):
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=False,
        dtype=dtype,
    )
    return prompt_embeds.to(device, dtype=dtype)


def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16

    os.makedirs(args.output_dir, exist_ok=True)

    pipe = CogVideoXPipeline.from_pretrained(args.model_id, torch_dtype=DTYPE).to(DEVICE)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    print(f"Generating baseline video for: {args.concept_prompt}")
    baseline_video = pipe(
        prompt=args.concept_prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6.0,
        generator=torch.Generator(device=DEVICE).manual_seed(args.seed),
    ).frames[0]
    baseline_path = os.path.join(args.output_dir, "baseline.mp4")
    export_to_video(baseline_video, baseline_path, fps=8)
    print(f"Baseline video saved to {baseline_path}")

    transformer = pipe.transformer
    transformer.train()
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)

    print("Pre-computing embeddings...")
    with torch.no_grad():
        concept_emb = encode_prompt(pipe, args.concept_prompt, DEVICE, DTYPE)
        null_emb = encode_prompt(pipe, "", DEVICE, DTYPE)

    del pipe.text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    scheduler = pipe.scheduler
    scheduler.set_timesteps(50)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(DEVICE)

    # Latent shape: 49 video frames → 13 latent frames; 480×720px → 60×90 latents
    batch_size = 1
    num_channels = 16
    num_latent_frames = 13
    latent_height = 60
    latent_width = 90

    saved_checkpoints = []

    print("Starting ESD training...")
    pbar = tqdm(range(args.steps))
    for step in pbar:
        optimizer.zero_grad()

        latents = torch.randn(
            (batch_size, num_channels, num_latent_frames, latent_height, latent_width),
            device=DEVICE,
            dtype=DTYPE,
        )

        target_timestep = np.random.choice(scheduler.timesteps.cpu().numpy())
        target_step = torch.tensor([target_timestep], dtype=torch.long, device=DEVICE)

        # Partially denoise to target timestep using frozen base model
        for t in scheduler.timesteps:
            t_tensor = torch.tensor([t], dtype=torch.long, device=DEVICE)
            if t_tensor <= target_step:
                break
            with torch.no_grad():
                with transformer.disable_adapter():
                    noise_pred = transformer(
                        hidden_states=latents.permute(0, 2, 1, 3, 4).to(dtype=DTYPE),
                        encoder_hidden_states=concept_emb.to(dtype=DTYPE),
                        timestep=t_tensor,
                    ).sample.permute(0, 2, 1, 3, 4)
                latents = scheduler.step(noise_pred, t_tensor, latents).prev_sample

        model_input = latents.to(dtype=DTYPE).permute(0, 2, 1, 3, 4)

        # Teacher predictions (frozen base model)
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
        target = model_pred_uncond - args.negative_guidance_scale * (model_pred_text - model_pred_uncond)

        # Student prediction (LoRA active)
        model_pred = transformer(
            hidden_states=model_input,
            encoder_hidden_states=concept_emb,
            timestep=target_step,
        ).sample

        loss = F.mse_loss(model_pred.float(), target.float())
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item():.4f}")

        if (step + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_step{step + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            transformer.save_pretrained(checkpoint_dir)
            saved_checkpoints.append((step + 1, checkpoint_dir))
            print(f"Checkpoint saved at step {step + 1}: {checkpoint_dir}")

    torch.cuda.empty_cache()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print("Training complete. Generating per-checkpoint videos...")
    for step_num, checkpoint_dir in saved_checkpoints:
        print(f"\nLoading checkpoint step {step_num}: {checkpoint_dir}")
        pipe = CogVideoXPipeline.from_pretrained(args.model_id, torch_dtype=DTYPE).to(DEVICE)
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, checkpoint_dir)
        pipe.transformer.eval()

        video = pipe(
            prompt=args.concept_prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6.0,
            generator=torch.Generator(device=DEVICE).manual_seed(args.seed),
        ).frames[0]

        video_path = os.path.join(args.output_dir, f"step{step_num}.mp4")
        export_to_video(video, video_path, fps=8)
        print(f"Video saved: {video_path}")

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    print("All done.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default="THUDM/CogVideoX-2b")
    parser.add_argument("--concept_prompt", type=str, required=True)
    parser.add_argument("--negative_guidance_scale", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Save checkpoint and generate video every N steps")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="outputs/quick_inspect")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
