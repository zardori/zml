#!/bin/bash
#SBATCH --job-name=quick_inspect
#SBATCH --output=logs/quick_inspect_%j.out
#SBATCH --error=logs/quick_inspect_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=4:00:00

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
export OUTPUT_DIR=outputs/quick_inspect_${TIMESTAMP}
mkdir -p "$OUTPUT_DIR"

python unlearn_quick_inspect.py \
    --model_id "${MODEL_ID:-THUDM/CogVideoX-5b}" \
    --concept_prompt "${CONCEPT_PROMPT:-A blue motorcycle}" \
    --negative_guidance_scale "${NEGATIVE_GUIDANCE_SCALE:-3.0}" \
    --steps "${STEPS:-20}" \
    --save_interval "${SAVE_INTERVAL:-1}" \
    --lora_rank "${LORA_RANK:-16}" \
    --lora_alpha "${LORA_ALPHA:-16.0}" \
    --learning_rate "${LR:-1e-3}" \
    --output_dir "$OUTPUT_DIR" \
    --seed "${SEED:-42}"
