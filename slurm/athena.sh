#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --partition=plgrid-gpu-a100

# Job name, time and log paths are supplied by submit_job.py as sbatch flags.
# CUDA ships with the PyTorch wheels installed by uv, so no `module load CUDA` is needed.

# Repo-dir guard for manual `sbatch slurm/athena.sh` runs; submit_job.py already cds into the repo.
if [ "$(basename "$PWD")" != zml ]; then
    echo "WARNING: for correct paths this script should be run from the 'zml' directory (main repo dir)."
    echo "  Current directory: $PWD"
    echo "Trying to guess repo dir based on username..."
    GUESSED_DIR="$PLG_GROUPS_STORAGE/plggtriplane/${USER:3}/zml"
    cd "$GUESSED_DIR" || { echo "Failed to change directory to guessed repo dir: $GUESSED_DIR. Exiting."; exit 1; }
    echo "Assumed $PWD as repo dir"
fi

export HF_HOME=hf_cache
mkdir -p "$HF_HOME"
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

case "${JOB_TYPE:-unlearn}" in
    unlearn)    uv run scripts/unlearn.py    --config "$CONFIG" --output_dir "$OUTPUT_DIR" ;;
    eval)       uv run scripts/eval.py       --config "$CONFIG" --output_dir "$OUTPUT_DIR" ;;
    precompute) uv run scripts/precompute.py --config "$CONFIG" --output_dir "$OUTPUT_DIR" ;;
    *) echo "Unknown JOB_TYPE: ${JOB_TYPE}" >&2; exit 1 ;;
esac
