"""Smoke-test method: no model loading, just verifies the full job pipeline works.

Checks imports, CUDA availability, logging (MLflow + W&B), and file output.
Each section is independent — a failure in one is recorded but doesn't stop the rest.
"""

import os
import sys
import traceback
from dataclasses import dataclass, field


@dataclass
class Config:
    output_dir: str


def _check(label: str, lines: list[str], failures: list[str]):
    """Context manager-style decorator isn't needed — just a helper to log section headers."""


def _run_section(label: str, fn, lines: list[str], failures: list[str]) -> None:
    try:
        fn(lines)
        lines.append(f"[OK] {label}")
    except Exception:
        msg = f"[FAIL] {label}:\n{traceback.format_exc().strip()}"
        lines.append(msg)
        failures.append(label)


def main(config: Config) -> None:
    lines: list[str] = []
    failures: list[str] = []

    def check_env(out: list[str]) -> None:
        out.append(f"Python: {sys.version}")

    def check_torch(out: list[str]) -> None:
        import torch
        out.append(f"PyTorch: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        out.append(f"CUDA available: {cuda_available}")
        if cuda_available:
            out.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
            out.append(f"CUDA version: {torch.version.cuda}")

    def check_diffusers(out: list[str]) -> None:
        import diffusers
        out.append(f"diffusers: {diffusers.__version__}")

    def check_transformers(out: list[str]) -> None:
        import transformers
        out.append(f"transformers: {transformers.__version__}")

    def check_mlflow(out: list[str]) -> None:
        import mlflow
        mlflow.log_metric("smoke/mlflow_ok", 1.0, step=0)
        out.append("mlflow logging OK")

    def check_wandb(out: list[str]) -> None:
        import wandb
        wandb.log({"smoke/wandb_ok": 1.0}, step=0)
        out.append("wandb logging OK")

    def check_file_output(out: list[str]) -> None:
        os.makedirs(config.output_dir, exist_ok=True)
        probe = os.path.join(config.output_dir, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        out.append(f"output_dir writable: {config.output_dir}")

    sections = [
        ("environment", check_env),
        ("torch + CUDA", check_torch),
        ("diffusers import", check_diffusers),
        ("transformers import", check_transformers),
        ("mlflow logging", check_mlflow),
        ("wandb logging", check_wandb),
        ("file output", check_file_output),
    ]

    for label, fn in sections:
        _run_section(label, fn, lines, failures)

    summary = "PASSED" if not failures else f"FAILED ({', '.join(failures)})"
    lines.append(f"\nSmoke test: {summary}")

    for line in lines:
        print(line)

    results_path = os.path.join(config.output_dir, "smoke_test_results.txt")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Results written to: {results_path}")
