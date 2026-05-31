#!/bin/bash
#SBATCH --job-name=gen_cog
#SBATCH --output=logs/gen_cog_%j.out
#SBATCH --error=logs/gen_cog_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=12:00:00

# From plg[nick] extract nick
CUT_USER=${USER:3}


if [ "$(basename "$PWD")" != zml ]; then
    echo "WARNING: for correct paths this script should be run from the 'zml' directory (main repo dir).
      Current directory: $PWD"
    echo "If your main repo dir has a different name, change it or include in the script check above as an alternative."
    echo "Trying to guess repo dir based on username..."
    GUESSED_DIR="$PLG_GROUPS_STORAGE/plggtriplane/$CUT_USER/zml"
    cd "$GUESSED_DIR" || { echo "Failed to change directory to guessed repo dir: $GUESSED_DIR. Exiting."; exit 1; }
    echo "Assumed $PWD as repo dir"
fi

# Load required modules
#module load Python/3.10.4
module load CUDA/12.0.0

# Activate virtual environment
#source "$PLG_GROUPS_STORAGE"/plggtriplane/btcaf/unlearning_env/bin/activate

# Prepare logs directory
mkdir -p logs

# Set Hugging Face / diffusers cache to group storage
export TRANSFORMERS_CACHE=$SCRATCH/hf/transformers
export SENTENCE_TRANSFORMERS_HOME=$SCRATCH/hf/st
export TORCH_HOME=$SCRATCH/torch
export HF_HOME=$SCRATCH/huggingface
export HF_TOKEN=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
export UV_CACHE_DIR=$SCRATCH/uv_cache

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export INPUT_DIR=outputs/experiment_baseline_20260419_181725

uv run zml/eval/check_for_fire.py \
    --input_dir "$INPUT_DIR" 
