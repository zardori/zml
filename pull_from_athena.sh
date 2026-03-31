#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="$(dirname "$0")/athena.conf"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: $CONFIG_FILE not found. Copy athena.conf.example to athena.conf and set your paths." >&2
    exit 1
fi
# shellcheck source=athena.conf.example
source "$CONFIG_FILE"

mkdir -p outputs logs

echo "Pulling outputs..."
rsync -avz --progress "${ATHENA_HOST}:${REMOTE_DIR}/outputs/" ./outputs/

echo "Pulling logs..."
rsync -avz --progress "${ATHENA_HOST}:${REMOTE_DIR}/logs/" ./logs/

echo "Done."
