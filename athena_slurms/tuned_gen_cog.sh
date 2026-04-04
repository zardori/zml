#!/bin/bash
#SBATCH --output=../logs/tuned_gen_cog_%j.out
#SBATCH --error=../logs/tuned_gen_cog_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=2:00:00

module load Python/3.10.4
module load CUDA/12.0.0

source $PLG_GROUPS_STORAGE/plggtriplane/btcaf/unlearning_env/bin/activate

export HF_HOME=$PLG_GROUPS_STORAGE/plggtriplane/poblos/hf_cache
mkdir -p $HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

export REPO_DIR="/net/pr2/projects/plgrid/plggtriplane/poblos/zml"

# Organise outputs per ngs variant and checkpoint name
CHECKPOINT_NAME=$(basename "$MODEL_CHECKPOINT")
export OUTPUT_DIR="$REPO_DIR/outputs/experiment_1/${NGS_VARIANT}/${CHECKPOINT_NAME}"
mkdir -p "$OUTPUT_DIR"

PROMPT_DIR="$REPO_DIR/prompts/vbench_prompts"
NUM_FRAMES=49
NUM_STEPS=50
GUIDANCE_SCALE=6.0
FPS=8
SEED=42

echo "Running checkpoint: $MODEL_CHECKPOINT"
echo "Output dir:         $OUTPUT_DIR"

python $REPO_DIR/generate_with_finetunned.py \
    --output_dir "$OUTPUT_DIR" \
    --prompt_dir "$PROMPT_DIR" \
    --num_frames $NUM_FRAMES \
    --num_inference_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --fps $FPS \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --seed $SEED