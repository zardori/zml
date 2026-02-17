#!/bin/bash
#SBATCH --job-name=check_for_nudity
#SBATCH --output=../logs/check_for_nudity_%j.out
#SBATCH --error=../logs/check_for_nudity_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=01:00:00

# -------------------------
# Load required modules
# -------------------------
module load Python/3.10.4
module load CUDA/12.0.0

# -------------------------
# Activate virtual environment
# -------------------------
source $PLG_GROUPS_STORAGE/plggtriplane/poblos/venv/bin/activate

# Set Hugging Face / diffusers cache to group storage
export HF_HOME=$PLG_GROUPS_STORAGE/plggtriplane/poblos/hf_cache
mkdir -p $HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

# -------------------------
# Define output directory
# -------------------------
export INPUT_DIR=$PLG_GROUPS_STORAGE/plggtriplane/poblos/finetunned_v2_cog/finetunned_v2_cog_outputs_our_promptset
mkdir -p $INPUT_DIR

# -------------------------
# Run the CogVideoX script
# -------------------------
# You can change the prompt and other parameters here


python /net/pr2/projects/plgrid/plggtriplane/poblos/check_for_nudity.py \
    --input_dir "$INPUT_DIR"