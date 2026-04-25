import os
import random
from dataclasses import dataclass

import mlflow
import wandb
import torch
import torch.nn.functional as F
import pandas as pd
from diffusers import CogVideoXPipeline
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from zml.unlearn.eval import evaluate
from zml.utils import set_seed


@dataclass
class Config:
    model_id: str
    prompts_path: str
    control_concept_prompts: str
    control_related_prompts: str
    control_unrelated_prompts: str
    preservation_prompts_path: str
    preservation_weight: float
    lora_rank: int
    lora_alpha: float
    negative_guidance_scale: float
    steps: int
    save_interval: int
    learning_rate: float
    lora_dropout: float
    output_dir: str
    eval_num_prompts: int
    eval_inference_steps: int
    global_seed: int | None = None


def _load_prompts_from_file(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main(config: Config) -> None:
    if config.global_seed is not None:
        set_seed(config.global_seed)

    control_concept_prompts = _load_prompts_from_file(config.control_concept_prompts)
    control_related_prompts = _load_prompts_from_file(config.control_related_prompts)
    control_unrelated_prompts = _load_prompts_from_file(config.control_unrelated_prompts)

    concept_prompts = pd.read_csv(config.prompts_path)["prompt"].tolist()
    preservation_prompts = pd.read_csv(config.preservation_prompts_path)["prompt"].tolist()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16

    pipe = CogVideoXPipeline.from_pretrained(config.model_id, torch_dtype=DTYPE).to(DEVICE)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    transformer = pipe.transformer
    transformer.train()
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=config.lora_dropout,
        bias="none",
    )

    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    pipe.transformer = transformer

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate)

    def encode_prompt(prompt: str) -> torch.Tensor:
        prompt_embeds, _ = pipe.encode_prompt(prompt=prompt, do_classifier_free_guidance=False)
        return prompt_embeds.to(DEVICE, dtype=DTYPE)

    scheduler = pipe.scheduler
    batch_size = 1
    num_channels = 16
    num_frames = 13
    height = 60
    width = 90
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps)

    latent_shape = (batch_size, num_channels, num_frames, height, width)

    print("Starting ESD with Preservation Training...")

    pbar = tqdm(range(config.steps))
    for step in pbar:
        concept_prompt = random.choice(concept_prompts)
        preserve_prompt = random.choice(preservation_prompts)

        with torch.no_grad():
            concept_emb = encode_prompt(concept_prompt)
            null_emb = encode_prompt("")
            preserve_emb = encode_prompt(preserve_prompt)

        optimizer.zero_grad()

        t_idx = random.randint(1, num_inference_steps - 1)
        t = scheduler.timesteps[t_idx]
        timesteps = t.unsqueeze(0).expand(batch_size).to(DEVICE)

        # Independent latents for forget and preserve steps avoid gradient coupling
        forget_latents = torch.randn(latent_shape, device=DEVICE, dtype=DTYPE)
        preserve_latents = torch.randn(latent_shape, device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            for ts in scheduler.timesteps:
                if ts <= t:
                    break
                ts_batch = ts.unsqueeze(0).expand(batch_size).to(DEVICE)

                forget_input = forget_latents.permute(0, 2, 1, 3, 4)
                forget_latents = scheduler.step(
                    transformer(hidden_states=forget_input, encoder_hidden_states=concept_emb, timestep=ts_batch).sample.permute(0, 2, 1, 3, 4),
                    ts,
                    forget_latents,
                ).prev_sample

                preserve_input = preserve_latents.permute(0, 2, 1, 3, 4)
                preserve_latents = scheduler.step(
                    transformer(hidden_states=preserve_input, encoder_hidden_states=preserve_emb, timestep=ts_batch).sample.permute(0, 2, 1, 3, 4),
                    ts,
                    preserve_latents,
                ).prev_sample

        forget_model_input = forget_latents.permute(0, 2, 1, 3, 4)
        preserve_model_input = preserve_latents.permute(0, 2, 1, 3, 4)

        # Teacher predictions (frozen base model, no LoRA)
        with torch.no_grad():
            with transformer.disable_adapter():
                pred_uncond = transformer(
                    hidden_states=forget_model_input,
                    encoder_hidden_states=null_emb,
                    timestep=timesteps,
                ).sample
                pred_concept = transformer(
                    hidden_states=forget_model_input,
                    encoder_hidden_states=concept_emb,
                    timestep=timesteps,
                ).sample
                esd_target = pred_uncond - config.negative_guidance_scale * (pred_concept - pred_uncond)

                teacher_preserve = transformer(
                    hidden_states=preserve_model_input,
                    encoder_hidden_states=preserve_emb,
                    timestep=timesteps,
                ).sample

        # Student predictions (trainable LoRA active)
        student_forget = transformer(
            hidden_states=forget_model_input,
            encoder_hidden_states=concept_emb,
            timestep=timesteps,
        ).sample

        student_preserve = transformer(
            hidden_states=preserve_model_input,
            encoder_hidden_states=preserve_emb,
            timestep=timesteps,
        ).sample

        loss_forget = F.mse_loss(student_forget.float(), esd_target.float())
        loss_preserve = F.mse_loss(student_preserve.float(), teacher_preserve.float())
        loss_total = loss_forget + config.preservation_weight * loss_preserve

        loss_total.backward()
        optimizer.step()

        mlflow.log_metric("train/loss_forget", loss_forget.item(), step=step)
        mlflow.log_metric("train/loss_preserve", loss_preserve.item(), step=step)
        mlflow.log_metric("train/loss_total", loss_total.item(), step=step)
        wandb.log(
            {
                "train/loss_forget": loss_forget.item(),
                "train/loss_preserve": loss_preserve.item(),
                "train/loss_total": loss_total.item(),
            },
            step=step,
        )
        pbar.set_description(f"forget={loss_forget.item():.4f} preserve={loss_preserve.item():.4f}")

        if (step + 1) % config.save_interval == 0:
            lora_output_dir = os.path.join(config.output_dir, f"cogvideox_erasure_lora_step{step + 1}")
            os.makedirs(lora_output_dir, exist_ok=True)
            transformer.save_pretrained(lora_output_dir)
            print(f"Checkpoint saved to: {lora_output_dir}")
            evaluate(
                pipe, transformer, config, step + 1,
                control_concept_prompts, control_related_prompts, control_unrelated_prompts,
            )

    print("Training Complete.")
