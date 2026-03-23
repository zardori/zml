#!/bin/bash
#SBATCH --output=../logs/collect_data_%j.out
#SBATCH --error=../logs/collect_data_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=12:00:00

module load Python/3.10.4
module load CUDA/12.0.0

source $PLG_GROUPS_STORAGE/plggtriplane/btcaf/unlearning_env/bin/activate

export HF_HOME=$PLG_GROUPS_STORAGE/plggtriplane/poblos/hf_cache
mkdir -p $HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

export REPO_DIR="/net/pr2/projects/plgrid/plggtriplane/poblos/zml"

cd $REPO_DIR
python generate_data.py