#!/bin/bash
#SBATCH --job-name=unlearn_cog
#SBATCH --output=../logs/unlearn_cog_%j.out
#SBATCH --error=../logs/unlearn_cog_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=14:00:00

# Load required modules
module load Python/3.10.4
module load CUDA/12.0.0

# Activate virtual environment
source $PLG_GROUPS_STORAGE/plggtriplane/btcaf/unlearning_env/bin/activate

# Set Hugging Face / diffusers cache to group storage
export HF_HOME=$PLG_GROUPS_STORAGE/plggtriplane/btcaf/hf_cache
mkdir -p $HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

# Specify parameters for unlearning
export REPO_DIR="/net/pr2/projects/plgrid/plggtriplane/poblos/zml"
cd $REPO_DIR

python unlearn_model.py \
    --prompts_path "$REPO_DIR/prompts/cogvideox_nudity.csv" \
    --lora_rank 8 \
    --lora_alpha 8.0 \
    --negative_guidance_scale 2.0 \
    --steps 1000 \
    --learning_rate 1e-3 \
    --lora_dropout 0.0 \
    --output_dir "$REPO_DIR/checkpoints"