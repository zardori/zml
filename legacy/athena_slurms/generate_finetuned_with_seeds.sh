#!/bin/bash
#SBATCH --job-name=gen_cog
#SBATCH --output=logs/gen_cog_%A_%a.out
#SBATCH --error=logs/gen_cog_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=8:00:00
#SBATCH --array=1-5

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

NUM_STEPS_FOR_CHECKPOINT=$((SLURM_ARRAY_TASK_ID * 200))
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
#export OUTPUT_DIR=outputs/experiment_finetuned_count${SLURM_ARRAY_TASK_ID}_${TIMESTAMP}
export OUTPUT_DIR=outputs/experiment_finetuned_weighted_steps_${NUM_STEPS_FOR_CHECKPOINT}_${TIMESTAMP}
mkdir -p "$OUTPUT_DIR"

# Specify parameters for generation
SEEDED_PROMPT_FILE=prompts/cogvideox_nudity.csv
NUM_FRAMES=49
NUM_STEPS=50
#MODEL_CHECKPOINT="/net/pr2/projects/plgrid/plggtriplane/btcaf/zml/outputs/unlearn_with_precomputed_latents_count${SLURM_ARRAY_TASK_ID}_20260316_234703/cogvideox_erasure_lora_nudity_step200"

MODEL_CHECKPOINT="/net/pr2/projects/plgrid/plggtriplane/zardori/zml/outputs/unlearn_with_weighted_step_sampling_20260324_205807/cogvideox_erasure_lora_nudity_step${NUM_STEPS_FOR_CHECKPOINT}"
GUIDANCE_SCALE=6.0
FPS=8

echo "Running array task $SLURM_ARRAY_TASK_ID with model checkpoint: $MODEL_CHECKPOINT"

python generate_with_finetunned.py \
    --output_dir "$OUTPUT_DIR" \
    --seeded_prompt_file "$SEEDED_PROMPT_FILE" \
    --num_frames $NUM_FRAMES \
    --num_inference_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --fps $FPS \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --n_prompts 15
