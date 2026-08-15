#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--cluster CLUSTER] [--experiment PATH] [--thread NAME] [--range A-B]"
    echo "          [--logs-only] [--include-weights] [--no-videos]"
    echo "  --cluster  Cluster name: athena or helios (reads cluster.conf, default: both)"
    echo "  --experiment PATH  Pull only this repo-relative experiment dir (e.g."
    echo "                     experiments/imagenet/exp066_split_chainsaw_dataset). Pair with --include-weights"
    echo "                     to fetch one experiment's latents without dragging every checkpoint in"
    echo "                     the repo across the wire. Skips the MLflow sync."
    echo "  --thread NAME      Pull only one thread, by its directory name under experiments/"
    echo "                     (imagenet, nudity, face_identity, shared). Skips the MLflow sync."
    echo "  --range A-B        Pull only experiments whose number is in [A, B] (inclusive), across every"
    echo "                     thread, or within --thread if given. A single number (--range 67) pulls one."
    echo "                     Skips the MLflow sync."
    echo "  --logs-only        Download only logs, skip experiment outputs"
    echo "  --include-weights  Include model weight files (.safetensors, .pt) when downloading outputs (excluded by default)"
    echo "  --no-videos        Skip generated video files (.mp4) when downloading outputs"
    exit 1
}

CLUSTER=""
EXPERIMENT=""
THREAD=""
RANGE=""
RANGE_START=0
RANGE_END=0
LOGS_ONLY=false
SKIP_ADAPTERS=true
SKIP_VIDEOS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster)          CLUSTER="$2"; shift ;;
        --experiment)       EXPERIMENT="${2%/}"; shift ;;
        --thread|--domain)  THREAD="${2%/}"; shift ;;
        --range)            RANGE="$2"; shift ;;
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
    if [[ -n "$THREAD" || -n "$RANGE" ]]; then
        echo "Error: --experiment already names one directory; drop --thread/--range." >&2
        exit 1
    fi
fi

if [[ -n "$THREAD" && ! "$THREAD" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Error: --thread must be a single directory name under experiments/ (got '${THREAD}')." >&2
    exit 1
fi

if [[ -n "$RANGE" ]]; then
    if [[ "$RANGE" =~ ^([0-9]+)(-([0-9]+))?$ ]]; then
        RANGE_START=$((10#${BASH_REMATCH[1]}))
        RANGE_END=$((10#${BASH_REMATCH[3]:-${BASH_REMATCH[1]}}))
    else
        echo "Error: --range must be N or N-M (got '${RANGE}')." >&2
        exit 1
    fi
    if (( RANGE_START > RANGE_END )); then
        echo "Error: --range start ${RANGE_START} is above its end ${RANGE_END}." >&2
        exit 1
    fi
fi

# Anything narrower than the full tree pulls a subtree only and leaves the MLflow sync alone.
NARROWED=false
[[ -n "$EXPERIMENT" || -n "$THREAD" || -n "$RANGE" ]] && NARROWED=true

if [[ "$NARROWED" == true && "$LOGS_ONLY" == true ]]; then
    echo "Error: --logs-only cannot be combined with --experiment/--thread/--range." >&2
    exit 1
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

# --thread / --range restrict which experiments a pull covers. The subtree narrows the rsync source
# (so a thread pull never lists the other threads at all); the filter rules pick the wanted
# experiment numbers out of whatever the subtree still contains. Rules are first-match-wins, so the
# directory rules that let rsync descend to depth 2 (thread/expNNN) and depth 3
# (archive/thread/expNNN) must precede the final catch-all exclude, and -m drops the traversed-only
# directories again so no empty thread folders appear locally.
SELECT_RELS=("experiments${THREAD:+/$THREAD}")
# A thread's retired experiments sit beside it in experiments/archive/<thread>/, so a thread pull
# has to name that subtree as well; the live threads have no archive counterpart yet, hence the test.
[[ -n "$THREAD" && -d "experiments/archive/${THREAD}" ]] && SELECT_RELS+=("experiments/archive/${THREAD}")
SELECT_FILTERS=()
if [[ -n "$RANGE" ]]; then
    SELECT_FILTERS=(-m)
    [[ -n "$THREAD" ]] || SELECT_FILTERS+=(--include='/*/' --include='/*/*/')
    for ((exp_id = RANGE_START; exp_id <= RANGE_END; exp_id++)); do
        printf -v exp_glob 'exp%03d_*' "$exp_id"
        if [[ -n "$THREAD" ]]; then
            SELECT_FILTERS+=(--include="/${exp_glob}/***")
        else
            SELECT_FILTERS+=(--include="/*/${exp_glob}/***" --include="/*/*/${exp_glob}/***")
        fi
    done
    SELECT_FILTERS+=(--exclude='*')
fi

if [[ -n "$EXPERIMENT" ]]; then
    SELECTION_LABEL="$EXPERIMENT"
else
    SELECTION_LABEL="${THREAD:+thread ${THREAD}}"
    if [[ -n "$RANGE" ]]; then
        SELECTION_LABEL="${SELECTION_LABEL:+${SELECTION_LABEL}, }$(printf 'exp%03d-exp%03d' "$RANGE_START" "$RANGE_END")"
    fi
fi

# Whether an experiment folder is covered by --thread/--range, for the transfers that name a folder
# directly instead of going through the filter rules above.
selected() {
    local name="$1" dest="$2"
    [[ -z "$THREAD" || "$dest" == */"${THREAD}"/* ]] || return 1
    [[ -n "$RANGE" ]] || return 0
    [[ "$name" =~ ^exp([0-9]+) ]] || return 1
    local id=$((10#${BASH_REMATCH[1]}))
    (( id >= RANGE_START && id <= RANGE_END ))
}

# Pull a peer's leftover pre-archive folders straight into their archive destination. One ssh
# lists which of them that repo root actually still has, so migrated roots cost one round trip.
pull_pre_archive() {
    local host="$1" rdir="$2"; shift 2
    local rsync_opts=("$@")
    [[ ${#ARCHIVE_DEST[@]} -gt 0 ]] || return 0

    local probe=() name stale
    for name in "${!ARCHIVE_DEST[@]}"; do
        selected "$name" "${ARCHIVE_DEST[$name]}" || continue
        probe+=("experiments/${name}")
    done
    [[ ${#probe[@]} -gt 0 ]] || return 0
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

# Mirror one repo-relative subtree from a single root. Only the member who ran an experiment has the
# folder, and a thread may not exist in every root either, so 23 (partial transfer, source missing)
# is the expected answer from the others rather than a failure.
pull_subtree() {
    local host="$1" rdir="$2" rel="$3"; shift 3
    local rsync_opts=("$@") rsync_exit=0

    mkdir -p "./${rel}"
    echo "  <- ${host}:${rdir}/${rel}/"
    rsync "${rsync_opts[@]}" "${host}:${rdir}/${rel}/" "./${rel}/" || rsync_exit=$?
    if [[ $rsync_exit -eq 23 ]]; then
        echo "  (not present in ${rdir}, skipping)"
        rmdir "./${rel}" 2>/dev/null || true  # undo the mkdir when nothing landed there
    elif [[ $rsync_exit -ne 0 ]]; then
        exit $rsync_exit
    fi
}

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
            echo "Pulling ${EXPERIMENT} from all members (${cluster})..."
            for rdir in "${remote_dirs[@]}"; do
                pull_subtree "$host" "$rdir" "$EXPERIMENT" "${rsync_opts[@]}"
            done
            return 0
        fi

        if [[ "$NARROWED" == true ]]; then
            # Thread and/or number range: one subtree per root, with the archived copies picked up
            # by the same filters and any unmigrated leftovers redirected as in the bulk pull.
            echo "Pulling ${SELECTION_LABEL} from all members (${cluster})..."
            local rel
            for rdir in "${remote_dirs[@]}"; do
                for rel in "${SELECT_RELS[@]}"; do
                    pull_subtree "$host" "$rdir" "$rel" "${rsync_opts[@]}" "${SELECT_FILTERS[@]}"
                done
                pull_pre_archive "$host" "$rdir" "${rsync_opts[@]}"
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
if [[ "$NARROWED" == true ]]; then
    echo "Pulling only ${SELECTION_LABEL}; skipping the MLflow sync."
fi

for cluster in "${CLUSTERS[@]}"; do
    pull_cluster "$cluster"
done

echo "Done."
