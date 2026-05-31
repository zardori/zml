#!/usr/bin/env -S uv run
"""Submit experiment to an HPC cluster, with optional grid search.

Usage:
    submit.py [--cluster CLUSTER] <slurm_script> <config>

Arguments:
    slurm_script   Path to SLURM script relative to remote dir (e.g. slurm/unlearn.sh)
    config         Path to experiment config YAML (e.g. experiments/exp001_esd_fire_lora8/config.yaml)

Options:
    --cluster      Cluster name; reads <cluster>.conf for connection details (default: athena)

Example:
    ./submit.py slurm/unlearn.sh experiments/exp001_esd_fire_lora8/config.yaml
    ./submit.py --cluster helios slurm/helios_unlearn.sh experiments/exp001_esd_fire_lora8/config.yaml

If the config contains any list-valued fields, a grid search is performed: one sbatch job
is submitted per combination in the Cartesian product of all list fields.
"""

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import yaml


SCRIPTS_DIR = Path(__file__).parent


def load_cluster_conf(cluster: str) -> dict[str, str]:
    conf_path = SCRIPTS_DIR / "cluster.conf"
    if not conf_path.exists():
        print(f"Error: {conf_path} not found. Copy cluster.conf.example to cluster.conf.", file=sys.stderr)
        sys.exit(1)
    script = f"""
source {shlex.quote(str(conf_path))}
case {shlex.quote(cluster)} in
    athena) echo "HOST=$ATHENA_HOST" && echo "REMOTE_DIR=$ATHENA_REMOTE_DIR" ;;
    helios) echo "HOST=$HELIOS_HOST" && echo "REMOTE_DIR=$HELIOS_REMOTE_DIR" ;;
    *) echo "Error: unknown cluster '{cluster}'" >&2; exit 1 ;;
esac
"""
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    conf: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, _, val = line.partition("=")
        conf[key.strip()] = val.strip()
    return conf


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


def submit_scalar(host: str, remote_dir: str, slurm_script: str, config_path: str, cluster: str) -> None:
    exp_dir = str(Path(config_path).parent)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = f"{exp_dir}/logs_{timestamp}"
    output_dir = f"{exp_dir}/outputs_{timestamp}"

    sbatch_cmd = (
        f"sbatch"
        f" --output={logs_dir}/unlearn_%j.out"
        f" --error={logs_dir}/unlearn_%j.err"
        f" --export=ALL,CONFIG={config_path},OUTPUT_DIR={output_dir}"
        f" {slurm_script}"
    )
    remote_cmd = f"cd {remote_dir} && mkdir -p {output_dir} {logs_dir} && git pull && {sbatch_cmd}"
    print(f"Submitting on {cluster}...")
    print(f"  Command: {sbatch_cmd}")
    subprocess.run(["ssh", host, remote_cmd], check=True)


def _write_config_and_submit(
    host: str,
    remote_dir: str,
    slurm_script: str,
    config_remote_path: str,
    output_dir: str,
    logs_dir: str,
    config_yaml: str,
) -> None:
    """Write an expanded config to remote and submit one sbatch job."""
    escaped = config_yaml.replace("'", "'\\''")
    write_cmd = f"mkdir -p $(dirname {config_remote_path}) {output_dir} {logs_dir} && printf '%s' '{escaped}' > {config_remote_path}"
    sbatch_cmd = (
        f"sbatch"
        f" --output={logs_dir}/unlearn_%j.out"
        f" --error={logs_dir}/unlearn_%j.err"
        f" --export=ALL,CONFIG={config_remote_path},OUTPUT_DIR={output_dir}"
        f" {slurm_script}"
    )
    remote_cmd = f"cd {remote_dir} && {write_cmd} && {sbatch_cmd}"
    subprocess.run(["ssh", host, remote_cmd], check=True)


def submit_grid(
    host: str,
    remote_dir: str,
    slurm_script: str,
    config_path: str,
    config: dict,
    cluster: str,
) -> None:
    combos = expand_grid(config)
    exp_dir = str(Path(config_path).parent)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_base = f"{exp_dir}/grid_{timestamp}"
    grid_keys = [k for k, v in config.items() if isinstance(v, list)]

    print(f"Grid search: {len(combos)} runs (Cartesian product of: {', '.join(grid_keys)})")
    for i, combo in enumerate(combos, start=1):
        varied = {k: combo[k] for k in grid_keys}
        print(f"  run_{i:03d}: {varied}")

    reply = input(f"\nSubmit all {len(combos)} jobs on {cluster}? [y/N] ").strip().lower()
    if reply != "y":
        print("Aborted.")
        sys.exit(1)

    print(f"\nPulling latest on {cluster}...")
    subprocess.run(["ssh", host, f"cd {remote_dir} && git pull"], check=True)

    for i, combo in enumerate(combos, start=1):
        run_dir = f"{grid_base}/run_{i:03d}"
        config_remote = f"{run_dir}/config.yaml"
        output_dir = f"{run_dir}/outputs"
        logs_dir = f"{run_dir}/logs"
        config_yaml = yaml.dump(combo, default_flow_style=False, sort_keys=False)

        _write_config_and_submit(
            host, remote_dir, slurm_script,
            config_remote, output_dir, logs_dir, config_yaml,
        )
        print(f"  run_{i:03d}: submitted")

    print(f"\nSubmitted {len(combos)} jobs. Grid configs and outputs: {grid_base}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit experiment to HPC cluster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cluster", default="athena", help="Cluster name (reads <cluster>.conf, default: athena)")
    parser.add_argument("slurm_script", help="Path to SLURM script relative to remote dir")
    parser.add_argument("config", help="Path to experiment config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    conf = load_cluster_conf(args.cluster)
    host = conf["HOST"]
    remote_dir = conf["REMOTE_DIR"]

    warnings = check_git_state()
    if warnings:
        print(f"Warning: you have {', '.join(warnings)}.")
        reply = input("Continue anyway? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            sys.exit(1)

    config = load_config(args.config)

    if any(isinstance(v, list) for v in config.values()):
        submit_grid(host, remote_dir, args.slurm_script, args.config, config, args.cluster)
    else:
        submit_scalar(host, remote_dir, args.slurm_script, args.config, args.cluster)


if __name__ == "__main__":
    main()
