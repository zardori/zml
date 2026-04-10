#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage: $0 <slurm_script> <config>

Arguments:
  slurm_script   Path to SLURM script (e.g. slurm/unlearn.sh)
  config         Path to experiment config YAML (e.g. experiments/exp001_esd_fire_lora8/config.yaml)

Example:
  $0 slurm/unlearn.sh experiments/exp001_esd_fire_lora8/config.yaml
EOF
    exit 1
}

[[ $# -ne 2 ]] && usage

SLURM_SCRIPT="$1"
CONFIG="$2"

CONFIG_FILE="$(dirname "$0")/athena.conf"
[[ -f "$CONFIG_FILE" ]] || { echo "Error: $CONFIG_FILE not found." >&2; exit 1; }
source "$CONFIG_FILE"

# Git state warnings
WARNINGS=()
if [[ -n $(git status --porcelain) ]]; then
    WARNINGS+=("uncommitted changes")
fi
if git rev-parse @{u} &>/dev/null; then
    UNPUSHED=$(git rev-list @{u}..HEAD --count)
    [[ $UNPUSHED -gt 0 ]] && WARNINGS+=("$UNPUSHED unpushed commit(s)")
fi
if [[ ${#WARNINGS[@]} -gt 0 ]]; then
    echo "Warning: you have $(IFS=', '; echo "${WARNINGS[*]}")."
    read -r -p "Continue anyway? [y/N] " REPLY
    [[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi

EXP_DIR=$(dirname "${CONFIG:?CONFIG env var is required (e.g. experiments/exp001_esd_fire_lora8/config.yaml)}")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGS_DIR="${EXP_DIR}/logs_${TIMESTAMP}"
OUTPUT_DIR="${EXP_DIR}/outputs_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

SBATCH_CMD=(sbatch --output="${LOGS_DIR}/unlearn_%j.out" --error="${LOGS_DIR}/unlearn_%j.err" --export=ALL,"CONFIG=${CONFIG}","OUTPUT_DIR=${OUTPUT_DIR}" "${SLURM_SCRIPT}")
REMOTE_CMD=$(printf '%q ' "${SBATCH_CMD[@]}")

echo "Submitting on Athena..."
echo "  Command: ${SBATCH_CMD[*]}"
ssh "${ATHENA_HOST}" "cd ${REMOTE_DIR} && git pull && ${REMOTE_CMD}"
