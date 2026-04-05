#!/bin/bash
#SBATCH --job-name=gen_cog
#SBATCH --output=../logs/gen_cog_%j.out
#SBATCH --error=../logs/gen_cog_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=2:00:00

# Load required modules
module load Python/3.10.4
module load CUDA/12.0.0

# Activate virtual environment
source $PLG_GROUPS_STORAGE/plggtriplane/btcaf/unlearning_env/bin/activate

# From plg[nick] extract nick
CUT_USER=${USER:3}

# Set Hugging Face / diffusers cache to group storage
export HF_HOME=$PLG_GROUPS_STORAGE/plggtriplane/$CUT_USER/hf_cache
mkdir -p $HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

# Specify parameters for generation
export REPO_DIR="/net/pr2/projects/plgrid/plggtriplane/$CUT_USER/zml"
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export OUTPUT_DIR=$REPO_DIR/outputs/experiment_baseline_${TIMESTAMP}
mkdir -p $OUTPUT_DIR

PROMPT_DIR="${PROMPT_DIR:-$REPO_DIR/prompts/vbench_prompts}"

python $REPO_DIR/generate_with_baseline.py \
    --output_dir "$OUTPUT_DIR" \
    --prompt_dir "$PROMPT_DIR" \
    --num_frames "${NUM_FRAMES:-49}" \
    --num_inference_steps "${NUM_STEPS:-30}" \
    --guidance_scale "${GUIDANCE_SCALE:-6.0}" \
    --fps "${FPS:-8}" \
    --seed "${SEED:-42}"