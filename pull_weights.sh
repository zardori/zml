#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <cluster> <remote-path>"
    echo "  cluster      Cluster name: athena or helios"
    echo "  remote-path  Path relative to the repo root on the cluster,"
    echo "               e.g. experiments/exp006_esd/outputs_20260419/cogvideox_lora_step1000"
    exit 1
}

if [[ $# -ne 2 ]]; then
    usage
fi

CLUSTER="$1"
REMOTE_PATH="${2#/}"  # strip any leading slash

CONFIG_FILE="$(dirname "$0")/cluster.conf"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: $CONFIG_FILE not found. Copy cluster.conf.example to cluster.conf and set your paths." >&2
    exit 1
fi
# shellcheck source=cluster.conf.example
source "$CONFIG_FILE"

case "$CLUSTER" in
    athena) host="$ATHENA_HOST"; remote_dirs=("${ATHENA_REMOTE_DIRS[@]}") ;;
    helios) host="$HELIOS_HOST"; remote_dirs=("${HELIOS_REMOTE_DIRS[@]}") ;;
    *) echo "Error: unknown cluster '${CLUSTER}'." >&2; exit 1 ;;
esac

mkdir -p "$(dirname "$REMOTE_PATH")"

found=false
for rdir in "${remote_dirs[@]}"; do
    echo "Trying ${host}:${rdir}/${REMOTE_PATH} ..."
    rsync_exit=0
    rsync -avz --progress "${host}:${rdir}/${REMOTE_PATH}/" "./${REMOTE_PATH}/" || rsync_exit=$?
    if [[ $rsync_exit -eq 0 ]]; then
        found=true
        break
    elif [[ $rsync_exit -eq 23 ]]; then
        echo "  (not found in ${rdir}, skipping)"
    else
        exit $rsync_exit
    fi
done

if [[ "$found" == false ]]; then
    echo "Error: '${REMOTE_PATH}' not found in any remote directory on ${CLUSTER}." >&2
    exit 1
fi

echo "Done."
