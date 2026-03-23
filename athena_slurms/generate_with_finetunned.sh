#!/bin/bash
#SBATCH --job-name=tuned_gen_cog
#SBATCH --output=../logs/tuned_gen_cog_%j.out
#SBATCH --error=../logs/tuned_gen_cog_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=0:30:00

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

PROMPT_DIR="$REPO_DIR/prompts/vbench_prompts"
NUM_FRAMES=49
NUM_STEPS=30
GUIDANCE_SCALE=6.0
FPS=8
MODEL_CHECKPOINT="/net/pr2/projects/plgrid/plggtriplane/zardori/zml/outputs/unlearn_with_precomputed_latents_20260303_231050/cogvideox_erasure_lora_nudity_step200"
SEED=42

python $REPO_DIR/generate_with_finetunned.py \
    --output_dir "$OUTPUT_DIR" \
    --prompt_dir "$PROMPT_DIR" \
    --num_frames $NUM_FRAMES \
    --num_inference_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --fps $FPS \
    --model_checkpoint $MODEL_CHECKPOINT \
    --seed $SEED