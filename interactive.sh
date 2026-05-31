#!/usr/bin/env bash
set -euo pipefail

CLUSTER="helios"
TIME="1:00:00"
GPUS="1"

usage() {
    echo "Usage: $0 [--cluster CLUSTER] [-t TIME] [-g GPUS]"
    echo "  --cluster  Cluster name: athena or helios (default: helios)"
    echo "  -t         Job time limit (default: 1:00:00)"
    echo "  -g         Number of GPUs (default: 1)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster) CLUSTER="$2"; shift ;;
        -t)        TIME="$2"; shift ;;
        -g)        GPUS="$2"; shift ;;
        -h)        usage ;;
        *)         usage ;;
    esac
    shift
done

CONFIG_FILE="$(dirname "$0")/cluster.conf"
[[ -f "$CONFIG_FILE" ]] || { echo "Error: $CONFIG_FILE not found. Copy cluster.conf.example to cluster.conf." >&2; exit 1; }
source "$CONFIG_FILE"

case "$CLUSTER" in
    athena) HOST="$ATHENA_HOST"; PARTITION="plgrid-gpu-a100"; ACCOUNT="plgunhype-gpu-a100" ;;
    helios) HOST="$HELIOS_HOST"; PARTITION="plgrid-gpu-gh200"; ACCOUNT="plgunhype-gpu-gh200" ;;
    *) echo "Error: unknown cluster '${CLUSTER}'. Add it to the case blocks in interactive.sh and cluster.conf.example." >&2; exit 1 ;;
esac

echo "Connecting to ${CLUSTER} compute node (partition: ${PARTITION}, gpu: ${GPUS}, time: ${TIME})..."
echo "Tip: once connected:"
echo "  module avail CUDA"
echo "  uv sync && python -c \"import torch; print(torch.cuda.is_available())\""
echo

SRUN_CMD="srun --partition=${PARTITION} --account=${ACCOUNT} --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem=32G --gres=gpu:${GPUS} --time=${TIME} --pty bash"
ssh -t "${HOST}" "${SRUN_CMD}"
