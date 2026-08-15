#!/bin/bash
# Locate a config's input paths on this cluster, run on a login node by submit_job.py.
#
# A config that names a file which does not exist on the cluster (typo, unfilled `outputs_TIMESTAMP`
# placeholder, data that was produced on the *other* cluster) otherwise fails only after the job
# starts, wasting a queue slot. This script performs the same lookup the entrypoints do at runtime
# (zml/paths.py): a repo-relative input counts as present if it is in this repo or in any peer's
# group-readable repo.
#
# Usage (from the repo root): slurm/check_config_paths.sh [--locate] [--config <path>] <cluster> <path>...
#   --config  the experiment config itself; checked in this repo only, since scripts/*.py open it
#             directly, without the peer fallback.
#   --locate  machine-readable mode for zml/cluster_sync.py: one tab-separated record per path on
#             stdout, and exit 0 whatever the outcome (the caller decides what to do about a miss,
#             which may be to fetch it from the other cluster).
#               FOUND<TAB><rel path><TAB><absolute path><TAB><size in bytes>
#               MISSING<TAB><rel path>
#               MISSING_CONFIG<TAB><path>
# Every path is checked before anything is reported, so one run lists all the mistakes.

set -uo pipefail

locate=0
config=""
while [ $# -gt 0 ]; do
    case "$1" in
        --locate) locate=1; shift ;;
        --config) config="$2"; shift 2 ;;
        *) break ;;
    esac
done

if [ $# -lt 1 ]; then
    echo "Usage: $0 [--locate] [--config <path>] <cluster> <path>..." >&2
    exit 2
fi

cluster="$1"
shift

ZML_CLUSTER="$cluster" source "$(dirname "$0")/peer_roots.sh"
IFS=: read -ra peer_roots <<< "$ZML_PEER_ROOTS"

# Echoes the first existing absolute path for a repo-relative one, this repo before the peers'.
locate_path() {
    local path="$1" root
    if [ -e "$path" ]; then
        echo "$PWD/$path"
        return 0
    fi
    for root in "${peer_roots[@]}"; do
        [ -n "$root" ] || continue
        if [ -e "$root/$path" ]; then
            echo "$root/$path"
            return 0
        fi
    done
    return 1
}

# Apparent size in bytes, so the caller can price a cross-cluster copy before starting it. Only
# asked for in --locate mode: on a network filesystem it walks the whole tree.
path_size() {
    local size
    size=$(du -sb -- "$1" 2>/dev/null | cut -f1)
    echo "${size:-0}"
}

# Kept apart because they fail for different reasons and need different advice.
missing=()
missing_config=()

if [ -n "$config" ] && [ ! -e "$config" ]; then
    missing_config+=("$config")
    [ "$locate" -eq 1 ] && printf 'MISSING_CONFIG\t%s\n' "$config"
fi

for path in "$@"; do
    if abs=$(locate_path "$path"); then
        if [ "$locate" -eq 1 ]; then
            printf 'FOUND\t%s\t%s\t%s\n' "$path" "$abs" "$(path_size "$abs")"
        elif [ "$abs" != "$PWD/$path" ]; then
            echo "  found in peer repo: $path -> $abs"
        fi
    else
        missing+=("$path")
        [ "$locate" -eq 1 ] && printf 'MISSING\t%s\n' "$path"
    fi
done

[ "$locate" -eq 1 ] && exit 0

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
