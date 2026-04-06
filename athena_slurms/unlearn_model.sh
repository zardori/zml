#!/bin/bash
#SBATCH --job-name=unlearn_cog
#SBATCH --output=./logs/unlearn_cog_%j.out
#SBATCH --error=./logs/unlearn_cog_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=24:00:00

# Load required modules
# module load Python/3.10.4
module load CUDA/12.0.0

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

# Activate virtual environment
source "$PLG_GROUPS_STORAGE"/plggtriplane/btcaf/zml/.venv/bin/activate

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

python unlearn_model.py \
    --model_id "${MODEL_ID:-THUDM/CogVideoX-5b}" \
    --prompts_path "${PROMPTS_PATH:-prompts/cogvideox_fire.csv}" \
    --control_related_prompts "${CONTROL_RELATED_PROMPTS:-prompts/cogvideox_fire_control_related.txt}" \
    --control_unrelated_prompts "${CONTROL_UNRELATED_PROMPTS:-prompts/cogvideox_fire_control_unrelated.txt}" \
    --lora_rank "${LORA_RANK:-8}" \
    --lora_alpha "${LORA_ALPHA:-8.0}" \
    --negative_guidance_scale "${NEGATIVE_GUIDANCE_SCALE:-2.0}" \
    --steps "${STEPS:-1000}" \
    --save_interval "${SAVE_INTERVAL:-200}" \
    --learning_rate "${LR:-1e-3}" \
    --lora_dropout "${LORA_DROPOUT:-0.0}" \
    --output_dir "$OUTPUT_DIR" \
    --eval_num_prompts "${EVAL_NUM_PROMPTS:-3}" \
    --eval_inference_steps "${EVAL_INFERENCE_STEPS:-50}"