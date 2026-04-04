#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--logs-only] [--skip-adapters]"
    echo "  --logs-only        Download only logs, skip outputs"
    echo "  --include-adapters Include .safetensors files when downloading outputs (excluded by default)"
    exit 1
}

LOGS_ONLY=false
SKIP_ADAPTERS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --logs-only)        LOGS_ONLY=true ;;
        --include-adapters) SKIP_ADAPTERS=false ;;
        *) usage ;;
    esac
    shift
done

CONFIG_FILE="$(dirname "$0")/athena.conf"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: $CONFIG_FILE not found. Copy athena.conf.example to athena.conf and set your paths." >&2
    exit 1
fi
# shellcheck source=athena.conf.example
source "$CONFIG_FILE"

mkdir -p outputs logs

if [[ "$LOGS_ONLY" == false ]]; then
    RSYNC_OPTS=(-avz --progress)
    if [[ "$SKIP_ADAPTERS" == true ]]; then
        RSYNC_OPTS+=(--exclude='*.safetensors' --exclude='adapter_config.json')
        echo "Skipping adapter files (*.safetensors, adapter_config.json). Use --include-adapters to download them."
    fi
    echo "Pulling outputs..."
    rsync "${RSYNC_OPTS[@]}" "${ATHENA_HOST}:${REMOTE_DIR}/outputs/" ./outputs/
fi

echo "Pulling logs..."
rsync -avz --progress "${ATHENA_HOST}:${REMOTE_DIR}/logs/" ./logs/

echo "Done."
