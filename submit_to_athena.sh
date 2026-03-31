#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <slurm_script> [extra sbatch args...]" >&2
    exit 1
fi

SLURM_SCRIPT="$1"; shift

CONFIG_FILE="$(dirname "$0")/athena.conf"
[[ -f "$CONFIG_FILE" ]] || { echo "Error: $CONFIG_FILE not found." >&2; exit 1; }
source "$CONFIG_FILE"

WARNINGS=()

if [[ -n $(git status --porcelain) ]]; then
    WARNINGS+=("uncommitted changes")
fi

# Only check for unpushed commits if a remote tracking branch exists
if git rev-parse @{u} &>/dev/null; then
    UNPUSHED=$(git rev-list @{u}..HEAD --count)
    if [[ $UNPUSHED -gt 0 ]]; then
        WARNINGS+=("$UNPUSHED unpushed commit(s)")
    fi
fi

if [[ ${#WARNINGS[@]} -gt 0 ]]; then
    echo "Warning: you have $(IFS=', '; echo "${WARNINGS[*]}")."
    read -r -p "Continue anyway? [y/N] " REPLY
    [[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi

echo "Submitting on Athena..."
ssh "${ATHENA_HOST}" "cd ${REMOTE_DIR} && git pull && sbatch athena_slurms/${SLURM_SCRIPT} $*"
