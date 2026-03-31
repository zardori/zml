#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="$(dirname "$0")/athena.conf"
[[ -f "$CONFIG_FILE" ]] || { echo "Error: $CONFIG_FILE not found." >&2; exit 1; }
source "$CONFIG_FILE"

USERS="plgzardori,plgbtcaf,plgpoblos"
INTERVAL=30

usage() {
    echo "Usage: $0 [-i interval_seconds]" >&2
    echo "  -i  Refresh interval in seconds (default: ${INTERVAL})" >&2
    exit 1
}

while getopts "i:h" opt; do
    case $opt in
        i) INTERVAL="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

watch_jobs() {
    clear
    echo "=== Athena SLURM jobs for: ${USERS} ==="
    echo "=== $(date) | refresh every ${INTERVAL}s | Ctrl+C to quit | run with -i [n_seconds] to change refresh rate ==="
    echo
    ssh "${ATHENA_HOST}" "squeue --users=${USERS} --format='%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R' 2>/dev/null || squeue -u ${USERS}"
}

tput smcup
trap 'tput rmcup' EXIT

while true; do
    watch_jobs
    sleep "${INTERVAL}"
done
