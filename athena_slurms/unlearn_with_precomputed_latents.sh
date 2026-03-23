#!/bin/bash
#SBATCH --job-name=unlearn_cog
#SBATCH --output=logs/unlearn_cog_%j.out
#SBATCH --error=logs/unlearn_cog_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=14:00:00

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
module load Python/3.10.4
module load CUDA/12.0.0

# Activate virtual environment
source "$PLG_GROUPS_STORAGE"/plggtriplane/btcaf/unlearning_env/bin/activate

# Prepare logs directory
mkdir -p logs

# Set Hugging Face / diffusers cache to group storage
export HF_HOME=hf_cache
mkdir -p "$HF_HOME"
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export OUTPUT_DIR=outputs/unlearn_with_precomputed_latents_${TIMESTAMP}
mkdir -p "$OUTPUT_DIR"

python unlearn_with_precomputed_latents.py \
    --metadata_file "$PLG_GROUPS_STORAGE/plggtriplane/poblos/zml/unlearning_dataset/metadata.json" \
    --metadata_count 5 \
    --latents_dir "$PLG_GROUPS_STORAGE/plggtriplane/poblos/zml/unlearning_dataset/latents" \
    --lora_rank 8 \
    --lora_alpha 8.0 \
    --negative_guidance_scale 2.0 \
    --steps 1000 \
    --learning_rate 1e-3 \
    --lora_dropout 0.0 \
    --output_dir "$OUTPUT_DIR"