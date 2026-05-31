#!/usr/bin/env bash
set -euo pipefail

INTERVAL=30

usage() {
    echo "Usage: $0 [-i interval_seconds]" >&2
    echo "  -i  Refresh interval in seconds (default: ${INTERVAL})" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i) INTERVAL="$2"; shift ;;
        -h) usage ;;
        *)  usage ;;
    esac
    shift
done

CONFIG_FILE="$(dirname "$0")/cluster.conf"
[[ -f "$CONFIG_FILE" ]] || { echo "Error: $CONFIG_FILE not found. Copy cluster.conf.example to cluster.conf." >&2; exit 1; }
source "$CONFIG_FILE"

watch_jobs() {
    local tmpfile_a tmpfile_h exit_a exit_h
    tmpfile_a=$(mktemp)
    tmpfile_h=$(mktemp)

    ssh "$ATHENA_HOST" "squeue --users=${SLURM_USERS} --format='%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R' 2>/dev/null || squeue -u ${SLURM_USERS}" >"$tmpfile_a" 2>&1 &
    local pid_a=$!
    ssh "$HELIOS_HOST" "squeue --users=${SLURM_USERS} --format='%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R' 2>/dev/null || squeue -u ${SLURM_USERS}" >"$tmpfile_h" 2>&1 &
    local pid_h=$!

    wait "$pid_a" && exit_a=0 || exit_a=1
    wait "$pid_h" && exit_h=0 || exit_h=1

    clear
    echo "=== SLURM jobs for: ${SLURM_USERS} ==="
    echo "=== $(date) | refresh every ${INTERVAL}s | Ctrl+C to quit | run with -i [n_seconds] to change refresh rate ==="
    echo

    echo "--- athena ---"
    if [[ $exit_a -eq 0 ]]; then
        cat "$tmpfile_a"
    else
        echo "(SSH/squeue failed — check connection to ${ATHENA_HOST}. Retrying in ${INTERVAL}s...)"
    fi
    echo

    echo "--- helios ---"
    if [[ $exit_h -eq 0 ]]; then
        cat "$tmpfile_h"
    else
        echo "(SSH/squeue failed — check connection to ${HELIOS_HOST}. Retrying in ${INTERVAL}s...)"
    fi

    rm -f "$tmpfile_a" "$tmpfile_h"
}

tput smcup
trap 'tput rmcup' EXIT

while true; do
    watch_jobs
    sleep "${INTERVAL}"
done
