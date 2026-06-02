#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--cluster CLUSTER] [--logs-only] [--include-weights]"
    echo "  --cluster  Cluster name: athena or helios (reads cluster.conf, default: athena)"
    echo "  --logs-only        Download only logs, skip experiment outputs"
    echo "  --include-weights  Include model weight files (.safetensors, .pt) when downloading outputs (excluded by default)"
    exit 1
}

CLUSTER="athena"
LOGS_ONLY=false
SKIP_ADAPTERS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster)          CLUSTER="$2"; shift ;;
        --logs-only)        LOGS_ONLY=true ;;
        --include-weights)  SKIP_ADAPTERS=false ;;
        *) usage ;;
    esac
    shift
done

CONFIG_FILE="$(dirname "$0")/cluster.conf"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: $CONFIG_FILE not found. Copy cluster.conf.example to cluster.conf and set your paths." >&2
    exit 1
fi
# shellcheck source=cluster.conf.example
source "$CONFIG_FILE"

case "$CLUSTER" in
    athena) HOST="$ATHENA_HOST"; REMOTE_DIR="$ATHENA_REMOTE_DIR"; REMOTE_DIRS=("${ATHENA_REMOTE_DIRS[@]}") ;;
    helios) HOST="$HELIOS_HOST"; REMOTE_DIR="$HELIOS_REMOTE_DIR"; REMOTE_DIRS=("${HELIOS_REMOTE_DIRS[@]}") ;;
    *) echo "Error: unknown cluster '${CLUSTER}'." >&2; exit 1 ;;
esac

mkdir -p experiments logs

if [[ "$LOGS_ONLY" == false ]]; then
    RSYNC_OPTS=(-avz --progress)
    if [[ "$SKIP_ADAPTERS" == true ]]; then
        RSYNC_OPTS+=(--exclude='*.safetensors' --exclude='*.pt' --exclude='adapter_config.json')
        echo "Skipping model weight files (*.safetensors, *.pt, adapter_config.json). Use --include-weights to download them."
    fi

    echo "Pulling experiment outputs from all members (${CLUSTER})..."
    for RDIR in "${REMOTE_DIRS[@]}"; do
        echo "  <- ${HOST}:${RDIR}/experiments/"
        rsync "${RSYNC_OPTS[@]}" "${HOST}:${RDIR}/experiments/" ./experiments/
    done
fi

echo "Pulling MLflow tracking data..."
for RDIR in "${REMOTE_DIRS[@]}"; do
    echo "  <- ${HOST}:${RDIR}/mlruns/"
    rsync_exit=0
    # Exit code 23 means partial transfer (e.g. source path missing) — safe to ignore
    rsync -avz --progress "${HOST}:${RDIR}/mlruns/" ./mlruns/ || rsync_exit=$?
    if [[ $rsync_exit -eq 23 ]]; then
        echo "  (no mlruns/ on ${CLUSTER} yet, skipping)"
    elif [[ $rsync_exit -ne 0 ]]; then
        exit $rsync_exit
    fi
done

echo "Done."
