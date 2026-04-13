#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--logs-only] [--include-adapters]"
    echo "  --logs-only        Download only logs, skip experiment outputs"
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

mkdir -p experiments logs

if [[ "$LOGS_ONLY" == false ]]; then
    RSYNC_OPTS=(-avz --progress)
    if [[ "$SKIP_ADAPTERS" == true ]]; then
        RSYNC_OPTS+=(--exclude='*.safetensors' --exclude='adapter_config.json')
        echo "Skipping adapter files (*.safetensors, adapter_config.json). Use --include-adapters to download them."
    fi

    echo "Pulling experiment outputs from all members..."
    for RDIR in "${REMOTE_DIRS[@]}"; do
        echo "  <- ${ATHENA_HOST}:${RDIR}/experiments/"
        rsync "${RSYNC_OPTS[@]}" "${ATHENA_HOST}:${RDIR}/experiments/" ./experiments/
    done
fi

echo "Pulling MLflow tracking data..."
for RDIR in "${REMOTE_DIRS[@]}"; do
    echo "  <- ${ATHENA_HOST}:${RDIR}/mlruns/"
    rsync_exit=0
    # Exit code 23 means partial transfer (e.g. source path missing) — safe to ignore
    rsync -avz --progress "${ATHENA_HOST}:${RDIR}/mlruns/" ./mlruns/ || rsync_exit=$?
    if [[ $rsync_exit -eq 23 ]]; then
        echo "  (no mlruns/ on athena yet, skipping)"
    elif [[ $rsync_exit -ne 0 ]]; then
        exit $rsync_exit
    fi
done

echo "Done."
