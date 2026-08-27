#!/usr/bin/env -S uv run
"""Submit experiment to an HPC cluster, with optional grid search.

Usage:
    submit_job.py <cluster> <config> [--slurm SLURM_SCRIPT] [--skip-path-check] [--no-fetch-missing] [--yes]

Arguments:
    cluster   Cluster name: athena or helios
    config    Path to experiment config YAML (e.g. experiments/nudity/exp062_frame_replace_nudity_eta2/config.yaml)

Options:
    --slurm   Path to SLURM script relative to remote dir; defaults to slurm/athena.sh
              for athena and slurm/helios.sh for helios
    --skip-path-check
              Submit even if config input paths are missing on the cluster (escape hatch for
              a job that produces its own inputs)
    --no-fetch-missing
              Do not offer to copy missing inputs from the other cluster
    --yes, -y Answer every confirmation with yes, for a non-interactive caller (the research
              agent). Nothing else changes: the path check still aborts a bad submission.

Example:
    ./submit_job.py athena experiments/nudity/exp062_frame_replace_nudity_eta2/config.yaml
    ./submit_job.py helios experiments/nudity/exp062_frame_replace_nudity_eta2/config.yaml
    ./submit_job.py helios experiments/nudity/exp062_frame_replace_nudity_eta2/config.yaml --slurm slurm/other.sh

Each config must set `slurm_time` (e.g. `slurm_time: 0-4:00:00`); it is passed as the sbatch
--time and there is no default. The optional `job_type` field (unlearn|eval|precompute, default
unlearn) selects the entrypoint and is exported to the SLURM script as JOB_TYPE.

If the config contains any list-valued fields, a grid search is performed: one sbatch job
is submitted per combination in the Cartesian product of all list fields.

Before submitting, the cluster repo is pulled and every repo-relative data path the config names is
checked to exist there — in your repo or in a peer's (slurm/check_config_paths.sh). Anything still
missing is looked for on the *other* cluster and, with your confirmation, copied over before the
job is submitted (zml/cluster_sync.py); an input nobody has anywhere aborts the submission instead
of failing the job minutes after it starts.

Once sbatch has accepted the jobs, the experiment's notes.md frontmatter is stamped
`status: active` and `submitted: <when> <cluster> jobs <ids>` — commit and push that, or the
registry (INDEX.md, the weekly report, the research agent) goes on reading a queued experiment as
one that was never submitted.
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import yaml

from zml.cluster_sync import (
    Cluster,
    ClusterSyncError,
    config_data_paths,
    fetch_missing_inputs,
    git_pull,
    load_cluster,
    locate_paths,
)
from tools.experiments_index import mark_submitted


CLUSTER_DEFAULT_SLURM: dict[str, str] = {
    "athena": "slurm/athena.sh",
    "helios": "slurm/helios.sh",
}

DEFAULT_JOB_TYPE = "unlearn"

NOTES_NAME = "notes.md"

SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")
# Last line of a successful submission, so a non-interactive caller can record what it launched
# without parsing sbatch's prose. Read by research_agent/zmlrepo.py.
JOB_IDS_PREFIX = "ZML_SUBMITTED_JOB_IDS="


def confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        print(f"{question} [auto-yes]")
        return True
    return input(f"{question} [y/N] ").strip().lower() == "y"


def run_sbatch(cluster: Cluster, remote_cmd: str) -> str | None:
    """Run one submission over ssh and return the job id sbatch reported.

    sbatch's line is echoed rather than swallowed, so an interactive run looks unchanged.
    """
    result = subprocess.run(
        ["ssh", cluster.host, remote_cmd], check=True, capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    match = SBATCH_JOB_ID_RE.search(result.stdout)
    return match.group(1) if match else None


def check_git_state() -> list[str]:
    warnings: list[str] = []
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if status.stdout.strip():
        warnings.append("uncommitted changes")
    upstream = subprocess.run(["git", "rev-parse", "@{u}"], capture_output=True, text=True)
    if upstream.returncode == 0:
        count = subprocess.run(
            ["git", "rev-list", "@{u}..HEAD", "--count"], capture_output=True, text=True
        )
        n = int(count.stdout.strip())
        if n > 0:
            warnings.append(f"{n} unpushed commit(s)")
    return warnings


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def remote_precheck(
    cluster: Cluster,
    paths: list[str],
    config_path: str | None,
    skip_path_check: bool,
    fetch_missing: bool,
    assume_yes: bool = False,
) -> None:
    """Pull the cluster repo and make sure the config's inputs are there; abort if they cannot be.

    Runs before any sbatch so a mistyped or never-uploaded input costs a second, not a queue slot
    and half an hour. The pull comes first because it may be what brings the inputs in; a path that
    is still absent is looked for on the other cluster, because the usual reason for a gap is that
    the data was produced over there and is nowhere else.
    """
    print(f"Pulling latest on {cluster.name}...")
    git_pull(cluster)

    located = locate_paths(cluster, paths, config=config_path)
    for rel, remote in located.found.items():
        if not remote.abs_path.startswith(f"{cluster.remote_dir}/"):
            print(f"  found in peer repo: {rel} -> {remote.abs_path}")

    missing = located.missing
    if missing and fetch_missing:
        missing = fetch_missing_inputs(cluster, missing, assume_yes=assume_yes)

    if not missing and not located.missing_config:
        n_checked = len(paths) + (1 if config_path else 0)
        print(f"Config paths OK ({n_checked} checked on {cluster.name}).")
        return

    print(
        f"ERROR: {len(missing) + len(located.missing_config)} config path(s) not available on "
        f"{cluster.name}:",
        file=sys.stderr,
    )
    for path in located.missing_config:
        print(f"  {path}  (not in your repo — committed and pushed?)", file=sys.stderr)
    for path in missing:
        print(f"  {path}", file=sys.stderr)

    if skip_path_check:
        print("Continuing despite the failed path check (--skip-path-check).")
        return
    print("Aborted: config path check failed. Re-run with --skip-path-check to submit anyway.",
          file=sys.stderr)
    sys.exit(1)


def expand_grid(config: dict) -> list[dict]:
    """Return all combinations from Cartesian product of list-valued fields."""
    grid_keys = [k for k, v in config.items() if isinstance(v, list)]
    scalars = {k: v for k, v in config.items() if not isinstance(v, list)}
    if not grid_keys:
        return [scalars]
    combos = []
    for combo in product(*[config[k] for k in grid_keys]):
        combos.append({**scalars, **dict(zip(grid_keys, combo))})
    return combos


def submit_scalar(
    cluster: Cluster,
    slurm_script: str,
    config_path: str,
    slurm_time: str,
    job_type: str,
) -> list[str]:
    exp_dir = str(Path(config_path).parent)
    job_name = Path(config_path).parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = f"{exp_dir}/logs_{timestamp}"
    output_dir = f"{exp_dir}/outputs_{timestamp}"

    sbatch_cmd = (
        f"sbatch"
        f" --job-name={job_name}"
        f" --time={slurm_time}"
        f" --output={logs_dir}/{job_type}_%j.out"
        f" --error={logs_dir}/{job_type}_%j.err"
        # ZML_TIME_LIMIT is what slurm/run_info.sh records when `scontrol` is unavailable on the
        # compute node (it is, on helios) — without it run_info.json cannot say how much headroom a
        # run had against its wall clock.
        f" --export=ALL,JOB_TYPE={job_type},CONFIG={config_path},OUTPUT_DIR={output_dir}"
        f",ZML_TIME_LIMIT={slurm_time}"
        f" {slurm_script}"
    )
    remote_cmd = f"cd {cluster.remote_dir} && mkdir -p {output_dir} {logs_dir} && {sbatch_cmd}"
    print(f"Submitting on {cluster.name}...")
    print(f"  Command: {sbatch_cmd}")
    job_id = run_sbatch(cluster, remote_cmd)
    return [job_id] if job_id else []


def _write_config_and_submit(
    cluster: Cluster,
    slurm_script: str,
    config_remote_path: str,
    output_dir: str,
    logs_dir: str,
    config_yaml: str,
    slurm_time: str,
    job_type: str,
    job_name: str,
) -> str | None:
    """Write an expanded config to remote and submit one sbatch job."""
    escaped = config_yaml.replace("'", "'\\''")
    write_cmd = f"mkdir -p $(dirname {config_remote_path}) {output_dir} {logs_dir} && printf '%s' '{escaped}' > {config_remote_path}"
    sbatch_cmd = (
        f"sbatch"
        f" --job-name={job_name}"
        f" --time={slurm_time}"
        f" --output={logs_dir}/{job_type}_%j.out"
        f" --error={logs_dir}/{job_type}_%j.err"
        f" --export=ALL,JOB_TYPE={job_type},CONFIG={config_remote_path},OUTPUT_DIR={output_dir}"
        f",ZML_TIME_LIMIT={slurm_time}"  # see the note in submit_scalar
        f" {slurm_script}"
    )
    remote_cmd = f"cd {cluster.remote_dir} && {write_cmd} && {sbatch_cmd}"
    return run_sbatch(cluster, remote_cmd)


def submit_grid(
    cluster: Cluster,
    slurm_script: str,
    config_path: str,
    config: dict,
    slurm_time: str,
    job_type: str,
    assume_yes: bool = False,
) -> list[str]:
    combos = expand_grid(config)
    exp_dir = str(Path(config_path).parent)
    exp_name = Path(config_path).parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_base = f"{exp_dir}/grid_{timestamp}"
    grid_keys = [k for k, v in config.items() if isinstance(v, list)]

    print(f"Grid search: {len(combos)} runs (Cartesian product of: {', '.join(grid_keys)})")
    for i, combo in enumerate(combos, start=1):
        varied = {k: combo[k] for k in grid_keys}
        print(f"  run_{i:03d}: {varied}")

    if not confirm(f"\nSubmit all {len(combos)} jobs on {cluster.name}?", assume_yes):
        print("Aborted.")
        sys.exit(1)

    job_ids: list[str] = []
    for i, combo in enumerate(combos, start=1):
        run_dir = f"{grid_base}/run_{i:03d}"
        config_remote = f"{run_dir}/config.yaml"
        output_dir = f"{run_dir}/outputs"
        logs_dir = f"{run_dir}/logs"
        config_yaml = yaml.dump(combo, default_flow_style=False, sort_keys=False)

        job_id = _write_config_and_submit(
            cluster, slurm_script,
            config_remote, output_dir, logs_dir, config_yaml,
            slurm_time=slurm_time, job_type=job_type, job_name=f"{exp_name}_run_{i:03d}",
        )
        if job_id:
            job_ids.append(job_id)
        print(f"  run_{i:03d}: submitted")

    print(f"\nSubmitted {len(combos)} jobs. Grid configs and outputs: {grid_base}/")
    return job_ids


def record_submission(config_path: str, cluster_name: str, job_ids: list[str]) -> None:
    """Stamp `status: active` and a `submitted:` line into the experiment's notes frontmatter.

    Without it an experiment stays `ready` with a "not run yet" takeaway for as long as it is in
    the queue, and every reader of the registry — INDEX.md, the weekly report, the research agent,
    which is given each experiment's frontmatter and not much else — reads that as never
    submitted. Manual submissions are exactly the ones nothing else would ever correct.

    Bookkeeping after the fact: the jobs are already queued, so notes that cannot be written are
    a warning to fix by hand, never a failure of a submission that has already happened.
    """
    notes = Path(config_path).parent / NOTES_NAME
    try:
        stamp = mark_submitted(notes, cluster_name, job_ids)
    except (OSError, ValueError) as exc:
        print(f"Warning: could not mark {notes} as submitted ({exc}) — set `status: active` there "
              f"by hand.", file=sys.stderr)
        return
    print(f"Marked {notes}: status active, submitted {stamp}.")
    print("  Commit and push it — the clusters and the research agent read it from the remote.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit experiment to HPC cluster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cluster", help="Cluster name: athena or helios")
    parser.add_argument("config", help="Path to experiment config YAML")
    parser.add_argument(
        "--slurm",
        default=None,
        help="Path to SLURM script relative to remote dir (default: slurm/athena.sh for athena, slurm/helios.sh for helios)",
    )
    parser.add_argument(
        "--skip-path-check",
        action="store_true",
        help="Submit even if config input paths are missing on the cluster",
    )
    parser.add_argument(
        "--no-fetch-missing",
        action="store_true",
        help="Do not offer to copy inputs that are missing here but present on the other cluster",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Answer every confirmation with yes (for non-interactive callers)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    slurm_script = args.slurm or CLUSTER_DEFAULT_SLURM.get(args.cluster)
    if slurm_script is None:
        print(f"Error: no default slurm script for cluster '{args.cluster}'; use --slurm", file=sys.stderr)
        sys.exit(1)

    try:
        cluster = load_cluster(args.cluster)
    except ClusterSyncError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    warnings = check_git_state()
    if warnings:
        print(f"Warning: you have {', '.join(warnings)}.")
        if not confirm("Continue anyway?", args.yes):
            print("Aborted.")
            sys.exit(1)

    config = load_config(args.config)
    slurm_time = config.pop("slurm_time", None)
    if not slurm_time:
        print(
            "Error: experiment config must set `slurm_time` (e.g. `slurm_time: 0-4:00:00`).",
            file=sys.stderr,
        )
        sys.exit(1)
    job_type = config.pop("job_type", DEFAULT_JOB_TYPE)

    is_grid = any(isinstance(v, list) for v in config.values())
    # A grid's per-run configs are written on the cluster, so only a scalar run reads a config
    # that has to be there already.
    try:
        remote_precheck(
            cluster,
            paths=config_data_paths(config),
            config_path=None if is_grid else args.config,
            skip_path_check=args.skip_path_check,
            fetch_missing=not args.no_fetch_missing,
            assume_yes=args.yes,
        )
    except ClusterSyncError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if is_grid:
        job_ids = submit_grid(cluster, slurm_script, args.config, config,
                              slurm_time=slurm_time, job_type=job_type, assume_yes=args.yes)
    else:
        job_ids = submit_scalar(cluster, slurm_script, args.config,
                                slurm_time=slurm_time, job_type=job_type)

    if job_ids:
        record_submission(args.config, cluster.name, job_ids)
    print(f"{JOB_IDS_PREFIX}{','.join(job_ids)}")


if __name__ == "__main__":
    main()
