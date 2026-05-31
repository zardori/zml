#!/bin/bash
#SBATCH --job-name=unlearn_cog
#SBATCH --output=../logs/unlearn_cog_%A_%a.out
#SBATCH --error=../logs/unlearn_cog_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=14:00:00
#SBATCH --array=0-5

# Load required modules
module load Python/3.10.4
module load CUDA/12.0.0

# Activate virtual environment
source $PLG_GROUPS_STORAGE/plggtriplane/btcaf/unlearning_env/bin/activate

# Set Hugging Face / diffusers cache to group storage
export HF_HOME=$PLG_GROUPS_STORAGE/plggtriplane/btcaf/hf_cache
mkdir -p $HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

# ── Parameter sweep ────────────────────────────────────────────────────────────
# negative_guidance_scale: (2, 4, 6)   → 3 values
# (lora_rank, lora_alpha) pairs: (8,8), (12,12) → 2 pairs
# Total: 3 × 2 = 6 jobs  (SLURM_ARRAY_TASK_ID: 0–5)

NEG_GUIDANCE_SCALES=(2.0 4.0 6.0)
LORA_RANKS=(8 12)
LORA_ALPHAS=(8.0 12.0)

# Decode flat index into the two axes
NGS_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))   # 0,1,2
LORA_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))  # 0,1

NEGATIVE_GUIDANCE_SCALE=${NEG_GUIDANCE_SCALES[$NGS_IDX]}
LORA_RANK=${LORA_RANKS[$LORA_IDX]}
LORA_ALPHA=${LORA_ALPHAS[$LORA_IDX]}

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  negative_guidance_scale = ${NEGATIVE_GUIDANCE_SCALE}"
echo "  lora_rank               = ${LORA_RANK}"
echo "  lora_alpha              = ${LORA_ALPHA}"

# ── Run ────────────────────────────────────────────────────────────────────────
export REPO_DIR="/net/pr2/projects/plgrid/plggtriplane/poblos/zml"
cd $REPO_DIR

python unlearn_model.py \
    --prompts_path "$REPO_DIR/prompts/cogvideox_nudity.csv" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --negative_guidance_scale ${NEGATIVE_GUIDANCE_SCALE} \
    --steps 1000 \
    --learning_rate 1e-3 \
    --lora_dropout 0.0 \
    --output_dir "$REPO_DIR/checkpoints/ngs${NEGATIVE_GUIDANCE_SCALE}_rank${LORA_RANK}"