#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--cluster CLUSTER] [--logs-only] [--include-weights]"
    echo "  --cluster  Cluster name: athena or helios (reads cluster.conf, default: both)"
    echo "  --logs-only        Download only logs, skip experiment outputs"
    echo "  --include-weights  Include model weight files (.safetensors, .pt) when downloading outputs (excluded by default)"
    exit 1
}

CLUSTER=""
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

if [[ -n "$CLUSTER" ]]; then
    CLUSTERS=("$CLUSTER")
else
    CLUSTERS=(athena helios)
fi

pull_cluster() {
    local cluster="$1" host remote_dirs
    case "$cluster" in
        athena) host="$ATHENA_HOST"; remote_dirs=("${ATHENA_REMOTE_DIRS[@]}") ;;
        helios) host="$HELIOS_HOST"; remote_dirs=("${HELIOS_REMOTE_DIRS[@]}") ;;
        *) echo "Error: unknown cluster '${cluster}'." >&2; exit 1 ;;
    esac

    if [[ "$LOGS_ONLY" == false ]]; then
        local rsync_opts=(-avz --progress)
        if [[ "$SKIP_ADAPTERS" == true ]]; then
            rsync_opts+=(--exclude='*.safetensors' --exclude='*.pt' --exclude='adapter_config.json')
        fi

        echo "Pulling experiment outputs from all members (${cluster})..."
        for rdir in "${remote_dirs[@]}"; do
            echo "  <- ${host}:${rdir}/experiments/"
            rsync "${rsync_opts[@]}" "${host}:${rdir}/experiments/" ./experiments/
        done
    fi

    echo "Pulling MLflow tracking data (${cluster})..."
    for rdir in "${remote_dirs[@]}"; do
        echo "  <- ${host}:${rdir}/mlruns/"
        local rsync_exit=0
        # Exit code 23 means partial transfer (e.g. source path missing) — safe to ignore
        rsync -avz --progress "${host}:${rdir}/mlruns/" ./mlruns/ || rsync_exit=$?
        if [[ $rsync_exit -eq 23 ]]; then
            echo "  (no mlruns/ on ${cluster} yet, skipping)"
        elif [[ $rsync_exit -ne 0 ]]; then
            exit $rsync_exit
        fi
    done
}

mkdir -p experiments logs

if [[ "$SKIP_ADAPTERS" == true && "$LOGS_ONLY" == false ]]; then
    echo "Skipping model weight files (*.safetensors, *.pt, adapter_config.json). Use --include-weights to download them."
fi

for cluster in "${CLUSTERS[@]}"; do
    pull_cluster "$cluster"
done

echo "Done."
