#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--cluster CLUSTER] [--experiment PATH] [--logs-only] [--include-weights] [--no-videos]"
    echo "  --cluster  Cluster name: athena or helios (reads cluster.conf, default: both)"
    echo "  --experiment PATH  Pull only this repo-relative experiment dir (e.g."
    echo "                     experiments/imagenet/exp066_split_chainsaw_dataset). Pair with --include-weights"
    echo "                     to fetch one experiment's latents without dragging every checkpoint in"
    echo "                     the repo across the wire. Skips the MLflow sync."
    echo "  --logs-only        Download only logs, skip experiment outputs"
    echo "  --include-weights  Include model weight files (.safetensors, .pt) when downloading outputs (excluded by default)"
    echo "  --no-videos        Skip generated video files (.mp4) when downloading outputs"
    exit 1
}

CLUSTER=""
EXPERIMENT=""
LOGS_ONLY=false
SKIP_ADAPTERS=true
SKIP_VIDEOS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster)          CLUSTER="$2"; shift ;;
        --experiment)       EXPERIMENT="${2%/}"; shift ;;
        --logs-only)        LOGS_ONLY=true ;;
        --include-weights)  SKIP_ADAPTERS=false ;;
        --no-videos)        SKIP_VIDEOS=true ;;
        *) usage ;;
    esac
    shift
done

# The path is spliced into both an ssh-side source and a local destination, so it has to stay inside
# the experiments tree — anything else would write outside the repo on a pull.
if [[ -n "$EXPERIMENT" ]]; then
    if [[ "$EXPERIMENT" != experiments/* || "$EXPERIMENT" == *..* ]]; then
        echo "Error: --experiment must be a repo-relative path under experiments/ (got '${EXPERIMENT}')." >&2
        exit 1
    fi
    if [[ "$LOGS_ONLY" == true ]]; then
        echo "Error: --experiment and --logs-only are mutually exclusive." >&2
        exit 1
    fi
fi

CONFIG_FILE="$(dirname "$0")/cluster.conf"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: $CONFIG_FILE not found. Copy cluster.conf.example to cluster.conf and set your paths." >&2
    exit 1
fi
# shellcheck source=cluster.conf.example
source "$CONFIG_FILE"

if [[ -n "$CLUSTER" ]]; then
    CLUSTERS=("$CLUSTER")
else
    CLUSTERS=(athena helios)
fi

cd "$(dirname "$0")"

# Archived experiments, by folder name -> the repo-relative path they now live at. A member
# who has not run tools/migrate_experiments.sh still holds their artifacts at the *pre-archive*
# path, and rsync cannot know the copy already sitting in experiments/archive/ is the same
# data — so a plain pull re-downloads all of it and re-creates the flat layout locally. We
# redirect those transfers into the archive path instead. Derived from the local archive tree,
# so it covers every experiment ever archived, not just the recent ones.
declare -A ARCHIVE_DEST=()
while IFS= read -r dir; do
    ARCHIVE_DEST["$(basename "$dir")"]="$dir"
done < <(find experiments/archive -mindepth 2 -maxdepth 2 -type d -name 'exp*' 2>/dev/null)

# Pull a peer's leftover pre-archive folders straight into their archive destination. One ssh
# lists which of them that repo root actually still has, so migrated roots cost one round trip.
pull_pre_archive() {
    local host="$1" rdir="$2"; shift 2
    local rsync_opts=("$@")
    [[ ${#ARCHIVE_DEST[@]} -gt 0 ]] || return 0

    local probe=() name stale
    for name in "${!ARCHIVE_DEST[@]}"; do
        probe+=("experiments/${name}")
    done
    stale=$(ssh "$host" "cd '${rdir}' 2>/dev/null && ls -d ${probe[*]} 2>/dev/null" || true)
    [[ -n "$stale" ]] || return 0

    while IFS= read -r remote_rel; do
        name="$(basename "$remote_rel")"
        echo "  <- ${rdir}/${remote_rel}/ (not migrated there) -> ${ARCHIVE_DEST[$name]}/"
        rsync "${rsync_opts[@]}" "${host}:${rdir}/${remote_rel}/" "./${ARCHIVE_DEST[$name]}/"
    done <<< "$stale"
}

# notes.md and config.yaml are source files we edit locally and push; the cluster only ever has
# whatever it last pulled, so letting rsync bring them back overwrites local edits with a stale
# copy. The exception is a grid's per-run config.yaml, which is generated on the cluster by
# submit_job.py and exists nowhere else. rsync applies filter rules first-match-wins, so the
# include must stay ahead of the excludes.
SOURCE_FILE_FILTERS=(
    --include='run_*/config.yaml'
    --exclude='config.yaml'
    --exclude='notes.md'
)

pull_cluster() {
    local cluster="$1" host remote_dirs
    case "$cluster" in
        athena) host="$ATHENA_HOST"; remote_dirs=("${ATHENA_REMOTE_DIRS[@]}") ;;
        helios) host="$HELIOS_HOST"; remote_dirs=("${HELIOS_REMOTE_DIRS[@]}") ;;
        *) echo "Error: unknown cluster '${cluster}'." >&2; exit 1 ;;
    esac

    if [[ "$LOGS_ONLY" == false ]]; then
        # -u (--update) matters because several roots hold copies of the same artifact with
        # different mtimes: a member who copied an experiment around without preserving times,
        # or an unmigrated root whose pre-archive folder maps onto the archive path another root
        # already provides. Without it those sources overwrite each other's mtime on every pull
        # and rsync re-transfers byte-identical files forever. Keeping the newest copy is a fixed
        # point, so the churn converges after one run.
        local rsync_opts=(-avzu --progress "${SOURCE_FILE_FILTERS[@]}")
        if [[ "$SKIP_ADAPTERS" == true ]]; then
            rsync_opts+=(--exclude='*.safetensors' --exclude='*.pt' --exclude='adapter_config.json')
        fi
        if [[ "$SKIP_VIDEOS" == true ]]; then
            rsync_opts+=(--exclude='*.mp4')
        fi

        if [[ -n "$EXPERIMENT" ]]; then
            # Narrow pull: one experiment, straight to its own path. The archive redirection is
            # irrelevant here because the caller named the destination themselves.
            mkdir -p "./${EXPERIMENT}"
            echo "Pulling ${EXPERIMENT} from all members (${cluster})..."
            for rdir in "${remote_dirs[@]}"; do
                echo "  <- ${host}:${rdir}/${EXPERIMENT}/"
                # Only the member who ran it has the folder; 23 (partial transfer, source missing)
                # is the expected answer from every other root.
                local exp_exit=0
                rsync "${rsync_opts[@]}" "${host}:${rdir}/${EXPERIMENT}/" "./${EXPERIMENT}/" || exp_exit=$?
                if [[ $exp_exit -eq 23 ]]; then
                    echo "  (not present in ${rdir}, skipping)"
                elif [[ $exp_exit -ne 0 ]]; then
                    exit $exp_exit
                fi
            done
            return 0
        fi

        # The bulk transfer never re-creates a flat folder for an archived experiment;
        # pull_pre_archive fetches those into experiments/archive/ instead.
        local bulk_opts=("${rsync_opts[@]}") name
        for name in "${!ARCHIVE_DEST[@]}"; do
            bulk_opts+=(--exclude="/${name}/")
        done

        echo "Pulling experiment outputs from all members (${cluster})..."
        for rdir in "${remote_dirs[@]}"; do
            echo "  <- ${host}:${rdir}/experiments/"
            rsync "${bulk_opts[@]}" "${host}:${rdir}/experiments/" ./experiments/
            pull_pre_archive "$host" "$rdir" "${rsync_opts[@]}"
        done
    fi

    echo "Pulling MLflow tracking data (${cluster})..."
    for rdir in "${remote_dirs[@]}"; do
        echo "  <- ${host}:${rdir}/mlruns/"
        local rsync_exit=0
        # Exit code 23 means partial transfer (e.g. source path missing) — safe to ignore
        rsync -avzu --progress "${host}:${rdir}/mlruns/" ./mlruns/ || rsync_exit=$?
        if [[ $rsync_exit -eq 23 ]]; then
            echo "  (no mlruns/ on ${cluster} yet, skipping)"
        elif [[ $rsync_exit -ne 0 ]]; then
            exit $rsync_exit
        fi
    done
}

mkdir -p experiments logs

if [[ "$SKIP_ADAPTERS" == true && "$LOGS_ONLY" == false ]]; then
    echo "Skipping model weight files (*.safetensors, *.pt, adapter_config.json). Use --include-weights to download them."
fi
if [[ "$SKIP_VIDEOS" == true && "$LOGS_ONLY" == false ]]; then
    echo "Skipping video files (*.mp4). Omit --no-videos to download them."
fi
if [[ "$LOGS_ONLY" == false ]]; then
    echo "Skipping experiment notes.md and config.yaml (kept in git; grid run_*/config.yaml still pulled)."
fi
if [[ -n "$EXPERIMENT" ]]; then
    echo "Pulling only ${EXPERIMENT}; skipping the MLflow sync."
fi

for cluster in "${CLUSTERS[@]}"; do
    pull_cluster "$cluster"
done

echo "Done."
