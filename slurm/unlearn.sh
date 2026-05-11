#!/bin/bash
#SBATCH --job-name=unlearn
#SBATCH --output=./logs/unlearn_%j.out
#SBATCH --error=./logs/unlearn_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=2-0:00:00

module load CUDA/12.0.0

if [ "$(basename "$PWD")" != zml ]; then
    echo "WARNING: for correct paths this script should be run from the 'zml' directory (main repo dir)."
    echo "  Current directory: $PWD"
    echo "If your main repo dir has a different name, change it or include in the script check above as an alternative."
    echo "Trying to guess repo dir based on username..."
    GUESSED_DIR="$PLG_GROUPS_STORAGE/plggtriplane/${USER:3}/zml"
    cd "$GUESSED_DIR" || { echo "Failed to change directory to guessed repo dir: $GUESSED_DIR. Exiting."; exit 1; }
    echo "Assumed $PWD as repo dir"
fi

mkdir -p logs

export HF_HOME=hf_cache
mkdir -p "$HF_HOME"
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

uv run scripts/unlearn.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR"
