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
#SBATCH --time=14:00:00

# -------------------------
# Load required modules
# -------------------------
module load Python/3.10.4
module load CUDA/12.0.0

# -------------------------
# Activate virtual environment
# -------------------------
source $PLG_GROUPS_STORAGE/plggtriplane/btcaf/unlearning_env/bin/activate

# Set Hugging Face / diffusers cache to group storage
export HF_HOME=$PLG_GROUPS_STORAGE/plggtriplane/btcaf/hf_cache
mkdir -p $HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

# -------------------------
# Define output directory
# -------------------------
# export OUTPUT_DIR=$PLG_GROUPS_STORAGE/plggtriplane/poblos/cogvideo_outputs_erased
# mkdir -p $OUTPUT_DIR

cd /net/pr2/projects/plgrid/plggtriplane/poblos/zml
python unlearn_model.py