#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=./logs/eval_%j.out
#SBATCH --error=./logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --time=0-4:00:00

if [ "$(basename "$PWD")" != zml ]; then
    echo "WARNING: for correct paths this script should be run from the 'zml' directory (main repo dir)."
    echo "  Current directory: $PWD"
    echo "Trying to guess repo dir based on username..."
    GUESSED_DIR="/net/scratch/hscra/plgrid/${USER}/zml"
    cd "$GUESSED_DIR" || { echo "Failed to change directory to guessed repo dir: $GUESSED_DIR. Exiting."; exit 1; }
    echo "Assumed $PWD as repo dir"
fi

mkdir -p logs

export HF_HOME=hf_cache
mkdir -p "$HF_HOME"
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

uv run scripts/eval.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR"
