#!/bin/bash
# Pre-submit check, run on a cluster login node by submit_job.py before any sbatch.
#
# A config that names a file which does not exist on the cluster (typo, unfilled `outputs_TIMESTAMP`
# placeholder, data that was moved) otherwise fails only after the job starts, wasting a queue slot.
# This script performs the same lookup the entrypoints do at runtime (zml/paths.py): a repo-relative
# input counts as present if it is in this repo or in any peer's group-readable repo.
#
# Usage (from the repo root): slurm/check_config_paths.sh [--config <path>] <cluster> <path>...
#   --config  the experiment config itself; checked in this repo only, since scripts/*.py open it
#             directly, without the peer fallback.
# Every path is checked before anything is reported, so one run lists all the mistakes.

set -uo pipefail

config=""
if [ "${1:-}" = "--config" ]; then
    config="$2"
    shift 2
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 [--config <path>] <cluster> <path>..." >&2
    exit 2
fi

cluster="$1"
shift

ZML_CLUSTER="$cluster" source "$(dirname "$0")/peer_roots.sh"
IFS=: read -ra peer_roots <<< "$ZML_PEER_ROOTS"

# Kept apart because they fail for different reasons and need different advice.
missing=()
missing_config=()

if [ -n "$config" ] && [ ! -e "$config" ]; then
    missing_config+=("$config")
fi

for path in "$@"; do
    [ -e "$path" ] && continue
    peer_hit=""
    for root in "${peer_roots[@]}"; do
        if [ -e "$root/$path" ]; then
            peer_hit="$root/$path"
            break
        fi
    done
    if [ -n "$peer_hit" ]; then
        echo "  found in peer repo: $path -> $peer_hit"
    else
        missing+=("$path")
    fi
done

n_missing=$(( ${#missing[@]} + ${#missing_config[@]} ))
n_checked=$#
[ -n "$config" ] && n_checked=$(( n_checked + 1 ))

if [ "$n_missing" -eq 0 ]; then
    echo "Config paths OK ($n_checked checked on $cluster)."
    exit 0
fi

echo "ERROR: $n_missing config path(s) not found on $cluster:" >&2
for path in "${missing_config[@]}"; do
    echo "  $path  (not in your repo — committed and pushed?)" >&2
done
for path in "${missing[@]}"; do
    echo "  $path" >&2
done
echo "Searched: $PWD ${peer_roots[*]}" >&2
exit 1
