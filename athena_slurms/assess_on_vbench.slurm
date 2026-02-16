#!/bin/bash
#SBATCH --job-name=cogvideox
#SBATCH --output=cogvideox_%j.out
#SBATCH --error=cogvideox_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=10:00:00

# -------------------------
# Load required modules
# -------------------------
module load Python/3.10.4
module load CUDA/12.0.0

# -------------------------
# Activate virtual environment
# -------------------------
source $PLG_GROUPS_STORAGE/plggtriplane/poblos/vbench_venv/bin/activate

# Set Hugging Face / diffusers cache to group storage
export HF_HOME=$PLG_GROUPS_STORAGE/plggtriplane/poblos/hf_cache
mkdir -p $HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

# -------------------------
# Define output directory
# -------------------------
export OUTPUT_DIR=$PLG_GROUPS_STORAGE/plggtriplane/poblos/cogvideo_outputs_vbench
mkdir -p $OUTPUT_DIR

# -------------------------
# Run the CogVideoX script
# -------------------------
# You can change the prompt and other parameters here
PROMPT_DIR="/net/pr2/projects/plgrid/plggtriplane/poblos/vbench_prompts"
NUM_FRAMES=49
NUM_STEPS=30
GUIDANCE_SCALE=6.0
FPS=8
MODEL_CHECKPOINT="/net/pr2/projects/plgrid/plggtriplane/poblos/cogvideox_erasure_lora_nudity"
SEED=42

vbench evaluate --videos_path $OUTPUT_DIR --dimension object_class