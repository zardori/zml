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

# Convert the START_TIME column from absolute SLURM timestamps to relative
# "time to start" strings (e.g. "1h30m", "45m02s", "~now").
# Uses substr() for fixed-position extraction (positions derived from the format
# string widths) and `date -d` for parsing — no gawk extensions required.
#
# Column positions in '%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %.20S %R':
#   1-106: everything before START_TIME, 107-126: START_TIME (20 chars), 127+: space+REASON
postprocess_start_times() {
    awk '
    BEGIN {
        "date +%s" | getline now
        close("date +%s")
    }
    NR==1 {
        print substr($0, 1, 106) sprintf("%20s", "TO_START") substr($0, 127)
        next
    }
    {
        ts = substr($0, 107, 20)
        gsub(/^ +/, "", ts)
        if (ts ~ /^[0-9]{4}-/) {
            cmd = "date -d \"" ts "\" +%s 2>/dev/null"
            cmd | getline target
            close(cmd)
            diff = target - now
            if (diff >= -60) {
                if (diff <= 0) val = "~now"
                else if (int(diff / 3600) > 0) val = sprintf("%dh%02dm", int(diff / 3600), int((diff % 3600) / 60))
                else val = sprintf("%dm%02ds", int(diff / 60), diff % 60)
                print substr($0, 1, 106) sprintf("%20s", val) substr($0, 127)
                next
            }
        }
        print
    }' "$1"
}

watch_jobs() {
    local tmpfile_a tmpfile_h exit_a exit_h
    tmpfile_a=$(mktemp)
    tmpfile_h=$(mktemp)

    ssh "$ATHENA_HOST" "squeue --users=${SLURM_USERS} --format='%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %.20S %R' 2>/dev/null || squeue -u ${SLURM_USERS}" >"$tmpfile_a" 2>&1 &
    local pid_a=$!
    ssh "$HELIOS_HOST" "squeue --users=${SLURM_USERS} --format='%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %.20S %R' 2>/dev/null || squeue -u ${SLURM_USERS}" >"$tmpfile_h" 2>&1 &
    local pid_h=$!

    wait "$pid_a" && exit_a=0 || exit_a=1
    wait "$pid_h" && exit_h=0 || exit_h=1

    clear
    echo "=== SLURM jobs for: ${SLURM_USERS} ==="
    echo "=== $(date) | refresh every ${INTERVAL}s | Ctrl+C to quit | run with -i [n_seconds] to change refresh rate ==="
    echo

    echo "--- athena ---"
    if [[ $exit_a -eq 0 ]]; then
        postprocess_start_times "$tmpfile_a"
    else
        echo "(SSH/squeue failed — check connection to ${ATHENA_HOST}. Retrying in ${INTERVAL}s...)"
    fi
    echo

    echo "--- helios ---"
    if [[ $exit_h -eq 0 ]]; then
        postprocess_start_times "$tmpfile_h"
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
