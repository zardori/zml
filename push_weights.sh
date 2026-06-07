#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <cluster> <local-path>"
    echo "  cluster     Cluster name: athena or helios"
    echo "  local-path  Local path to directory to push (relative to repo root),"
    echo "              e.g. experiments/exp006_esd/outputs_20260419/cogvideox_lora_step1000"
    exit 1
}

if [[ $# -ne 2 ]]; then
    usage
fi

CLUSTER="$1"
LOCAL_PATH="${2#./}"  # strip leading ./ if present
LOCAL_PATH="${LOCAL_PATH%/}"  # strip trailing slash

CONFIG_FILE="$(dirname "$0")/cluster.conf"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: $CONFIG_FILE not found. Copy cluster.conf.example to cluster.conf and set your paths." >&2
    exit 1
fi
# shellcheck source=cluster.conf.example
source "$CONFIG_FILE"

case "$CLUSTER" in
    athena) host="$ATHENA_HOST"; rdir="$ATHENA_REMOTE_DIR" ;;
    helios) host="$HELIOS_HOST"; rdir="$HELIOS_REMOTE_DIR" ;;
    *) echo "Error: unknown cluster '${CLUSTER}'." >&2; exit 1 ;;
esac

if [[ ! -d "$LOCAL_PATH" ]]; then
    echo "Error: local directory '${LOCAL_PATH}' does not exist." >&2
    exit 1
fi

remote_parent="${rdir}/$(dirname "$LOCAL_PATH")"

echo "Pushing ./${LOCAL_PATH}/ -> ${host}:${rdir}/${LOCAL_PATH}/"
ssh "$host" "mkdir -p '${remote_parent}'"
rsync -avz --progress "./${LOCAL_PATH}/" "${host}:${rdir}/${LOCAL_PATH}/"

echo "Done."
