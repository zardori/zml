#!/bin/bash
#SBATCH --job-name=unlearning_eval
#SBATCH --output=cogvideox_%j.out
#SBATCH --error=cogvideox_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=01:00:00

# -------------------------
# Load required modules
# -------------------------
module load Python/3.10.4
module load CUDA/12.0.0

# -------------------------
# Activate virtual environment
# -------------------------
source $PLG_GROUPS_STORAGE/plggtriplane/poblos/venv/bin/activate

# Set Hugging Face / diffusers cache to group storage
export HF_HOME=$PLG_GROUPS_STORAGE/plggtriplane/zardori/hf_cache
mkdir -p $HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

# -------------------------
# Define output directory
# -------------------------
export OUTPUT_DIR=$PLG_GROUPS_STORAGE/plggtriplane/zardori/cogvideo_outputs
mkdir -p $OUTPUT_DIR

# -------------------------
# Run the CogVideoX script
# -------------------------
# You can change the prompt and other parameters here
PROMPT="A blue motorcycle"
NUM_FRAMES=49
NUM_STEPS=30
GUIDANCE_SCALE=6.0
FPS=8

python /net/pr2/projects/plgrid/plggtriplane/zardori/cogvideo_script.py \
    --output_dir "$OUTPUT_DIR" \
    --prompt "$PROMPT" \
    --num_frames $NUM_FRAMES \
    --num_inference_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --fps $FPS