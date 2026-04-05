#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage: $0 <slurm_script> [options] [extra sbatch args...]

Options:
  --set KEY=VALUE ...   Pass env var overrides to the job script (space-separated list)
  --time HH:MM:SS       Override #SBATCH --time
  --mem SIZE            Override #SBATCH --mem  (e.g. 64G)
  --job-name NAME       Override #SBATCH --job-name
  --cpus-per-task N     Override #SBATCH --cpus-per-task
  --nodes N             Override #SBATCH --nodes

Examples:
  $0 unlearn_with_precomputed_latents.sh --set LORA_RANK=12 NEGATIVE_GUIDANCE_SCALE=4.0
  $0 unlearn_quick_inspect.sh --set STEPS=50 "CONCEPT_PROMPT=A cat on a bike" --time 2:00:00
  $0 unlearn_model.sh --set LORA_RANK=8 LR=5e-4 --job-name my_exp
EOF
    exit 1
}

[[ $# -lt 1 ]] && usage

SLURM_SCRIPT="$1"; shift

CONFIG_FILE="$(dirname "$0")/athena.conf"
[[ -f "$CONFIG_FILE" ]] || { echo "Error: $CONFIG_FILE not found." >&2; exit 1; }
source "$CONFIG_FILE"

# Parse arguments
EXPORT_VARS=()
SBATCH_OVERRIDES=()
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --set)
            shift
            [[ $# -gt 0 ]] || { echo "Error: --set requires at least one KEY=VALUE argument" >&2; exit 1; }
            # Consume all following KEY=VALUE pairs (stop at next flag or non-kv arg)
            while [[ $# -gt 0 && "$1" != --* && "$1" == *=* ]]; do
                EXPORT_VARS+=("$1"); shift
            done ;;
        --time|--mem|--job-name|--cpus-per-task|--nodes)
            [[ $# -gt 1 ]] || { echo "Error: $1 requires a value" >&2; exit 1; }
            SBATCH_OVERRIDES+=("$1=$2"); shift 2 ;;
        --time=*|--mem=*|--job-name=*|--cpus-per-task=*|--nodes=*)
            SBATCH_OVERRIDES+=("$1"); shift ;;
        -h|--help) usage ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

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

# Build sbatch command as an array for safe quoting
SBATCH_CMD=(sbatch)

if [[ ${#EXPORT_VARS[@]} -gt 0 ]]; then
    EXPORT_STR=$(IFS=','; echo "${EXPORT_VARS[*]}")
    SBATCH_CMD+=("--export=ALL,${EXPORT_STR}")
fi

for override in "${SBATCH_OVERRIDES[@]+"${SBATCH_OVERRIDES[@]}"}"; do
    SBATCH_CMD+=("$override")
done
for arg in "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"; do
    SBATCH_CMD+=("$arg")
done
SBATCH_CMD+=("athena_slurms/${SLURM_SCRIPT}")

# Serialize to a properly quoted string for remote execution
REMOTE_CMD=$(printf '%q ' "${SBATCH_CMD[@]}")

echo "Submitting on Athena..."
echo "  Command: ${SBATCH_CMD[*]}"
ssh "${ATHENA_HOST}" "cd ${REMOTE_DIR} && git pull && ${REMOTE_CMD}"
