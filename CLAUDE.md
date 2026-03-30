# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing **machine unlearning for CogVideoX** (a video diffusion transformer), specifically erasing nudity concepts using the ESD (Erased Stable Diffusion) method with LoRA fine-tuning. It runs on PLGrid HPC infrastructure via SLURM.

## Environment Setup

```bash
# Install dependencies using uv
uv sync

# Or with pip
pip install -r requirements.txt
```

Python 3.12 is required (see `.python-version`).

## Running Scripts

All main scripts are at the repository root and accept CLI arguments:

```bash
# Generate videos with the base (unmodified) model
python generate_with_baseline.py --prompts_file prompts/our_promptset.txt --output_dir outputs/baseline --num_inference_steps 50 --seed 42

# Generate videos with a fine-tuned LoRA model
python generate_with_finetunned.py --model_checkpoint checkpoints/step_200 --prompts_file prompts/our_promptset.txt --output_dir outputs/finetuned --limit_prompts 10

# Run unlearning (basic ESD)
python unlearn_model.py

# Run unlearning with concept preservation
python unlearn_model_preserve.py

# Run unlearning with pre-computed latents
python unlearn_with_precomputed_latents.py

# Pre-compute latent trajectories for training
python generate_data.py

# Check for nudity in generated videos
python benchmarks/check_for_nudity.py
```

## Submitting SLURM Jobs (PLGrid/Athena)

SLURM scripts are in `athena_slurms/`. Use `sbatch` to submit:

```bash
sbatch athena_slurms/unlearn_model.sh
sbatch athena_slurms/generate_baseline_with_seeds.sh
sbatch athena_slurms/generate_finetuned_with_seeds.sh   # array job over checkpoints
sbatch athena_slurms/unlearn_model_gridsearch.sh        # hyperparameter sweep (6 configs)
sbatch athena_slurms/unlearn_with_precomputed_latents.sh
```

## Architecture

### Core Workflow

1. **Data prep** (`generate_data.py`): Loads prompts, encodes with CogVideoX-5b, saves latent trajectories at each diffusion timestep + metadata JSON. Enables efficient training without re-encoding.

2. **Unlearning** (multiple scripts): Loads CogVideoX-5b transformer, injects LoRA adapters via PEFT, runs ESD training loop:
   - Unconditional noise prediction (teacher)
   - Concept noise prediction (to erase)
   - Optional: preservation prediction to maintain other capabilities
   - Saves LoRA checkpoints every 200 steps to `checkpoints/`

3. **Generation** (`generate_with_baseline.py`, `generate_with_finetunned.py`): Loads prompts from CSV or TXT, generates MP4 videos, saves to output directory named after the task.

4. **Evaluation** (`benchmarks/check_for_nudity.py`): Uses NudeNet to detect nudity across video frames, computes normalized detection rates. Also uses VBench prompts from `prompts/vbench_prompts/`.

### Unlearning Script Variants

| Script | Description |
|--------|-------------|
| `unlearn_model.py` | Basic ESD with LoRA |
| `unlearn_model_preserve.py` | ESD + concept preservation loss |
| `unlearn_with_precomputed_latents.py` | Uses pre-computed latents; supports `uniform` and `weighted` (temperature-based) timestep sampling strategies |

### Key Parameters

- **Base model**: `THUDM/CogVideoX-5b` (transformer)
- **LoRA rank/alpha**: grid-searched over `[8, 12]`
- **Negative guidance scale**: grid-searched over `[2.0, 4.0, 6.0]`
- **Checkpoint interval**: every 200 training steps
- **HPC resources**: 1× A100 GPU, 128 GB RAM, 16 CPUs, 8–14h wall-time

### Directory Conventions

- `outputs/` — generated videos (gitignored), subdirectory name matches the task/run
- `checkpoints/` — saved LoRA adapters (gitignored)
- `logs/` — SLURM job logs (gitignored)
- `prompts/` — prompt CSV/TXT files used for generation and training
- `athena_slurms/` — SLURM job scripts for HPC
- `benchmarks/` — evaluation utilities
