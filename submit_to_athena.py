#!/usr/bin/env -S uv run
"""Submit experiment to Athena HPC cluster, with optional grid search.

Usage:
    submit_to_athena.py <slurm_script> <config>

Arguments:
    slurm_script   Path to SLURM script relative to remote dir (e.g. slurm/unlearn.sh)
    config         Path to experiment config YAML (e.g. experiments/exp001_esd_fire_lora8/config.yaml)

Example:
    ./submit_to_athena.py slurm/unlearn.sh experiments/exp001_esd_fire_lora8/config.yaml

If the config contains any list-valued fields, a grid search is performed: one sbatch job
is submitted per combination in the Cartesian product of all list fields.
"""

import shlex
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import yaml


ATHENA_CONF = Path(__file__).parent / "athena.conf"


def load_athena_conf(conf_path: Path) -> dict[str, str]:
    script = f'source {shlex.quote(str(conf_path))} && echo "ATHENA_HOST=$ATHENA_HOST" && echo "REMOTE_DIR=$REMOTE_DIR"'
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


def submit_scalar(athena_host: str, remote_dir: str, slurm_script: str, config_path: str) -> None:
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
    print(f"Submitting on Athena...")
    print(f"  Command: {sbatch_cmd}")
    subprocess.run(["ssh", athena_host, remote_cmd], check=True)



def _write_config_and_submit(
    athena_host: str,
    remote_dir: str,
    slurm_script: str,
    config_remote_path: str,
    output_dir: str,
    logs_dir: str,
    config_yaml: str,
) -> str:
    """Write an expanded config to remote, submit one sbatch job, return the job output."""
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
    subprocess.run(["ssh", athena_host, remote_cmd], check=True)


def submit_grid(
    athena_host: str,
    remote_dir: str,
    slurm_script: str,
    config_path: str,
    config: dict,
) -> None:
    combos = expand_grid(config)
    exp_dir = str(Path(config_path).parent)
    grid_base = f"{exp_dir}/grid"
    grid_keys = [k for k, v in config.items() if isinstance(v, list)]

    print(f"Grid search: {len(combos)} runs (Cartesian product of: {', '.join(grid_keys)})")
    for i, combo in enumerate(combos, start=1):
        varied = {k: combo[k] for k in grid_keys}
        print(f"  run_{i:03d}: {varied}")

    reply = input(f"\nSubmit all {len(combos)} jobs? [y/N] ").strip().lower()
    if reply != "y":
        print("Aborted.")
        sys.exit(1)

    print("\nPulling latest on Athena...")
    subprocess.run(["ssh", athena_host, f"cd {remote_dir} && git pull"], check=True)

    for i, combo in enumerate(combos, start=1):
        run_dir = f"{grid_base}/run_{i:03d}"
        config_remote = f"{run_dir}/config.yaml"
        output_dir = f"{run_dir}/outputs"
        logs_dir = f"{run_dir}/logs"
        config_yaml = yaml.dump(combo, default_flow_style=False, sort_keys=False)

        job_output = _write_config_and_submit(
            athena_host, remote_dir, slurm_script,
            config_remote, output_dir, logs_dir, config_yaml,
        )
        print(f"  run_{i:03d}: {job_output}")

    print(f"\nSubmitted {len(combos)} jobs. Grid configs and outputs: {grid_base}/")


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: submit_to_athena.py <slurm_script> <config>\n\n"
            "  slurm_script   Path to SLURM script (e.g. slurm/unlearn.sh)\n"
            "  config         Path to experiment config YAML (e.g. experiments/exp001_esd_fire_lora8/config.yaml)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    slurm_script = sys.argv[1]
    config_path = sys.argv[2]

    if not ATHENA_CONF.exists():
        print(f"Error: {ATHENA_CONF} not found.", file=sys.stderr)
        sys.exit(1)

    conf = load_athena_conf(ATHENA_CONF)
    athena_host = conf["ATHENA_HOST"]
    remote_dir = conf["REMOTE_DIR"]

    warnings = check_git_state()
    if warnings:
        print(f"Warning: you have {', '.join(warnings)}.")
        reply = input("Continue anyway? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            sys.exit(1)

    config = load_config(config_path)

    if any(isinstance(v, list) for v in config.values()):
        submit_grid(athena_host, remote_dir, slurm_script, config_path, config)
    else:
        submit_scalar(athena_host, remote_dir, slurm_script, config_path)


if __name__ == "__main__":
    main()
