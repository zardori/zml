#!/usr/bin/env bash
set -euo pipefail

CLUSTER="athena"
INTERVAL=30

usage() {
    echo "Usage: $0 [--cluster CLUSTER] [-i interval_seconds]" >&2
    echo "  --cluster  Cluster name: athena or helios (reads cluster.conf, default: athena)" >&2
    echo "  -i         Refresh interval in seconds (default: ${INTERVAL})" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster) CLUSTER="$2"; shift ;;
        -i)        INTERVAL="$2"; shift ;;
        -h)        usage ;;
        *)         usage ;;
    esac
    shift
done

CONFIG_FILE="$(dirname "$0")/cluster.conf"
[[ -f "$CONFIG_FILE" ]] || { echo "Error: $CONFIG_FILE not found. Copy cluster.conf.example to cluster.conf." >&2; exit 1; }
source "$CONFIG_FILE"

case "$CLUSTER" in
    athena) HOST="$ATHENA_HOST" ;;
    helios) HOST="$HELIOS_HOST" ;;
    *) echo "Error: unknown cluster '${CLUSTER}'." >&2; exit 1 ;;
esac

watch_jobs() {
    clear
    echo "=== ${CLUSTER} SLURM jobs for: ${SLURM_USERS} ==="
    echo "=== $(date) | refresh every ${INTERVAL}s | Ctrl+C to quit | run with -i [n_seconds] to change refresh rate ==="
    echo
    if ! ssh "${HOST}" "squeue --users=${SLURM_USERS} --format='%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R' 2>/dev/null || squeue -u ${SLURM_USERS}"; then
        echo "(SSH/squeue failed — check connection to ${HOST}. Retrying in ${INTERVAL}s...)"
    fi
}

tput smcup
trap 'tput rmcup' EXIT

while true; do
    watch_jobs
    sleep "${INTERVAL}"
done
