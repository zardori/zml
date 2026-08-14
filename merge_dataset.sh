#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--cluster athena|helios] --output DIR --source METADATA_FILE LATENTS_DIR [--source ...]"
    echo "  --cluster  Cluster to run the merge on (reads cluster.conf, default: helios)"
    echo "  --output   Repo-relative output dir for the merged dataset (e.g."
    echo "             experiments/exp116_split_face_obama_dataset_scaleup/combined_dataset)"
    echo "  --source   Repeatable: one metadata.json + latents_dir pair per source dataset."
    echo
    echo "Runs zml/precompute/merge_frame_replace_datasets.py on the cluster login node, where the"
    echo "precomputed .pt latents actually live (pull_results.sh excludes them by default, and"
    echo "combined_dataset/ is gitignored -- it has to be built where it will be read from)."
    exit 1
}

CLUSTER="helios"
OUTPUT=""
SOURCE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster) CLUSTER="$2"; shift 2 ;;
        --output)  OUTPUT="$2"; shift 2 ;;
        --source)
            [[ $# -ge 3 ]] || usage
            SOURCE_ARGS+=("--source" "$2" "$3")
            shift 3
            ;;
        -h|--help) usage ;;
        *) usage ;;
    esac
done

[[ -n "$OUTPUT" ]] || usage
[[ ${#SOURCE_ARGS[@]} -gt 0 ]] || usage

# These strings are spliced into a remote shell command below, so keep them inside the repo the
# same way pull_results.sh guards --experiment.
check_repo_relative() {
    local path="$1" label="$2"
    if [[ "$path" == /* || "$path" == *..* ]]; then
        echo "Error: $label must be a repo-relative path with no '..' (got '$path')." >&2
        exit 1
    fi
}
check_repo_relative "$OUTPUT" "--output"
for ((i = 0; i < ${#SOURCE_ARGS[@]}; i += 3)); do
    check_repo_relative "${SOURCE_ARGS[$((i + 1))]}" "--source metadata file"
    check_repo_relative "${SOURCE_ARGS[$((i + 2))]}" "--source latents dir"
done

CONFIG_FILE="$(dirname "$0")/cluster.conf"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: $CONFIG_FILE not found. Copy cluster.conf.example to cluster.conf and set your paths." >&2
    exit 1
fi
# shellcheck source=cluster.conf.example
source "$CONFIG_FILE"

case "$CLUSTER" in
    athena) HOST="$ATHENA_HOST"; REMOTE_DIR="$ATHENA_REMOTE_DIR" ;;
    helios) HOST="$HELIOS_HOST"; REMOTE_DIR="$HELIOS_REMOTE_DIR" ;;
    *) echo "Error: unknown cluster '${CLUSTER}'." >&2; exit 1 ;;
esac

cd "$(dirname "$0")"

# The cluster only ever sees what was pushed -- warn, don't block, same as submit_job.py's
# check_git_state (this step doesn't submit a job, so there's no queue slot to waste, but a merge
# built from stale filtered-metadata JSONs is still worth flagging).
if [[ -n "$(git status --porcelain)" ]]; then
    echo "WARNING: uncommitted local changes -- the cluster will merge whatever was last pushed."
fi
if git rev-parse '@{u}' >/dev/null 2>&1; then
    unpushed=$(git rev-list '@{u}..HEAD' --count)
    if [[ "$unpushed" -gt 0 ]]; then
        echo "WARNING: ${unpushed} unpushed commit(s) -- the cluster will merge whatever was last pushed."
    fi
fi

# Build the remote command with every argument individually quoted. Uses the login node's own
# stock python3, not `uv run` or the repo's .venv: login nodes have no uv, and the .venv (built by
# `uv run` on a compute node) has wheels for the GPU nodes' architecture, which need not match the
# login node's (helios: aarch64 GH200 vs. x86_64 login node) -- it may not even execute there.
# merge_frame_replace_datasets.py and zml/paths.py are stdlib-only and deliberately avoid PEP 585
# annotation syntax so they run on helios' login node Python (3.6) unmodified.
remote_cmd="cd $(printf '%q' "$REMOTE_DIR") && git pull"
remote_cmd+=" && ZML_CLUSTER=$(printf '%q' "$CLUSTER") source slurm/peer_roots.sh"
remote_cmd+=" && python3 -m zml.precompute.merge_frame_replace_datasets"
for ((i = 0; i < ${#SOURCE_ARGS[@]}; i++)); do
    remote_cmd+=" $(printf '%q' "${SOURCE_ARGS[$i]}")"
done
remote_cmd+=" --output_dir $(printf '%q' "$OUTPUT")"

echo "Merging on ${CLUSTER} (${HOST}:${REMOTE_DIR})..."
ssh "$HOST" "$remote_cmd"

echo "Done. ${OUTPUT} is gitignored (built fresh from the sources above) -- the git-tracked record"
echo "of what went in is the --source metadata files, not this output directory."
