#!/bin/bash
# Record where a job ran and how long it took, as $OUTPUT_DIR/run_info.json.
#
# Sourced by slurm/{athena,helios}.sh with ZML_CLUSTER set, before the JOB_TYPE dispatch.
#
# Why this exists: nothing else in the repo answers "how long does this kind of run take, and on
# which cluster" in a way you can look up. `runtime.json` is written by the eval entrypoint only, and
# only on success; `summary.json` covers unlearn runs; precompute writes neither. So sizing
# `slurm_time` for a new experiment meant grepping timestamps out of SLURM logs — and a job that
# *timed out* left nothing at all, which is exactly the case you most need the number for (exp065
# died at 163/200 videos against a 10 h limit with no record of it).
#
# The record is written twice: once at startup (outcome "running", so an in-flight job is visible),
# then rewritten from an EXIT trap so a crash or a wall-clock kill still lands on disk.
#
# Outcome is inferred from the exit status. SLURM kills an over-time job with SIGTERM, so 143 (and
# 137 for a SIGKILL that follows) reads as "timeout" — note that a manual `scancel` is
# indistinguishable, hence `time_limit` is recorded alongside for a human to compare against.

zml_json_escape() {
    printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

# Emit a JSON field; a literal `null` when the value is empty, otherwise a quoted string.
zml_json_field() {
    local key="$1" value="$2" trailing="${3-,}"
    if [ -z "$value" ]; then
        printf '  "%s": null%s\n' "$key" "$trailing"
    else
        printf '  "%s": "%s"%s\n' "$key" "$(zml_json_escape "$value")" "$trailing"
    fi
}

zml_write_run_info() {
    local outcome="$1" exit_code="$2" ended_at="" elapsed=""

    if [ "$outcome" != "running" ]; then
        ended_at="$(date -Iseconds)"
        elapsed=$(( $(date +%s) - ZML_RUN_START_EPOCH ))
    fi

    mkdir -p "$OUTPUT_DIR" 2>/dev/null || return 0
    {
        printf '{\n'
        zml_json_field cluster    "${ZML_RUN_CLUSTER:-}"
        zml_json_field job_id     "${SLURM_JOB_ID:-}"
        zml_json_field job_name   "${SLURM_JOB_NAME:-}"
        zml_json_field job_type   "${JOB_TYPE:-unlearn}"
        zml_json_field node       "${SLURMD_NODENAME:-$(hostname)}"
        zml_json_field config     "${CONFIG:-}"
        zml_json_field output_dir "${OUTPUT_DIR:-}"
        zml_json_field git_sha    "${ZML_RUN_GIT_SHA:-}"
        zml_json_field time_limit "${ZML_RUN_TIME_LIMIT:-}"
        zml_json_field started_at "${ZML_RUN_STARTED_AT:-}"
        zml_json_field ended_at   "$ended_at"
        printf '  "elapsed_s": %s,\n' "${elapsed:-null}"
        printf '  "exit_code": %s,\n' "${exit_code:-null}"
        zml_json_field outcome "$outcome" ""
        printf '}\n'
    } > "${OUTPUT_DIR}/run_info.json" 2>/dev/null || true
}

zml_on_exit() {
    local exit_code=$?
    local outcome
    case "$exit_code" in
        0)        outcome="completed" ;;
        143|137)  outcome="timeout" ;;
        *)        outcome="failed" ;;
    esac
    zml_write_run_info "$outcome" "$exit_code"
}

# Bash does not propagate a caught signal into the EXIT trap's $? on its own — without this, a job
# killed at its wall clock records exit_code 0 / "completed", the exact lie this file exists to
# prevent. Re-exiting with 128+SIGTERM makes the EXIT trap see it.
zml_on_term() {
    exit 143
}

# `ZML_CLUSTER=helios source slurm/run_info.sh` is a *temporary* assignment: it is gone by the time
# the EXIT trap runs. Copy it into a variable that survives.
ZML_RUN_CLUSTER="${ZML_CLUSTER:-}"
ZML_RUN_START_EPOCH="$(date +%s)"
ZML_RUN_STARTED_AT="$(date -Iseconds)"
ZML_RUN_GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null)"
# SLURM exposes no env var for the wall clock, so ask `scontrol` — but it is not reachable from
# helios' compute nodes (every helios run_info.json so far has time_limit null), so fall back to the
# value submit_job.py exports from the config's `slurm_time`.
ZML_RUN_TIME_LIMIT="$(scontrol show job "${SLURM_JOB_ID:-}" -o 2>/dev/null \
    | tr ' ' '\n' | sed -n 's/^TimeLimit=//p')"
ZML_RUN_TIME_LIMIT="${ZML_RUN_TIME_LIMIT:-${ZML_TIME_LIMIT:-}}"
export ZML_RUN_CLUSTER ZML_RUN_START_EPOCH ZML_RUN_STARTED_AT ZML_RUN_GIT_SHA ZML_RUN_TIME_LIMIT

zml_write_run_info running ""
trap zml_on_term TERM
trap zml_on_exit EXIT
