import contextlib
import json
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import mlflow
import wandb
import yaml

from zml.eval.eval_model import Config, main as eval_main
from zml.eval.generate_videos import GenerateConfig, main as generate_main
from zml.paths import resolve_config_paths


def _write_runtime(output_dir: str, start_time: float, started_at: str) -> None:
    """Record wall-clock runtime next to the eval outputs (synced by pull_results.sh),
    so slurm_time for future eval jobs can be calibrated from measured runs."""
    elapsed_hours = (time.time() - start_time) / 3600.0
    runtime = {
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_hours": round(elapsed_hours, 3),
    }
    with open(Path(output_dir) / "runtime.json", "w") as f:
        json.dump(runtime, f, indent=2)
    print(f"Eval wall-clock time: {elapsed_hours:.2f} h")


def run_eval(params: dict, config_path: str, output_dir: str, experiment_name: str,
             config_cls=Config, eval_fn=eval_main) -> None:
    # `disable_mlflow` is read (not popped) so `Config` still receives it via **params.
    disable_mlflow = params.get("disable_mlflow", False)

    if not disable_mlflow:
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(experiment_name)

    # The eval module logs all its metrics to mlflow/wandb itself; the entrypoint only owns the run
    # lifecycle (and the runtime record, written even when the job dies).
    with (contextlib.nullcontext() if disable_mlflow else mlflow.start_run()):
        if not disable_mlflow:
            mlflow.log_params(params)
            mlflow.log_artifact(config_path)
        try:
            wandb.init(
                project="zml",
                entity="zardori-zml",
                name=experiment_name,
                config=params,
            )
            wandb.save(config_path)
        except Exception as e:
            print(f"WARNING: wandb init failed ({e}), continuing without W&B tracking.")
            wandb.init(mode="disabled")
        start_time = time.time()
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            eval_fn(config_cls(**params, output_dir=output_dir))
        finally:
            # Written even on failure so a partial/killed job still leaves its runtime on disk.
            _write_runtime(output_dir, start_time, started_at)
        wandb.finish()


def run_generate(params: dict, output_dir: str) -> None:
    # Plain generation produces no metrics, so the mlflow/wandb run lifecycle is skipped.
    generate_main(GenerateConfig(**params, output_dir=output_dir))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory for outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        params = resolve_config_paths(yaml.safe_load(f))

    config_path = Path(args.config)
    if config_path.parent.parent.name == "grid":
        experiment_name = config_path.parent.parent.parent.name
    else:
        experiment_name = config_path.parent.name

    params.pop("slurm_time", None)  # infra key, not an eval param
    params.pop("job_type", None)  # infra key, selects the entrypoint; not an eval param
    # `mode` is dispatch metadata that neither Config accepts, so it is popped here.
    mode = params.pop("mode", "eval")
    if mode == "generate":
        run_generate(params, args.output_dir)
    elif mode == "eval":
        run_eval(params, args.config, args.output_dir, experiment_name)
    elif mode == "imagenet":
        # Imported here so the fire/nudity eval paths keep no import-time dependency on the
        # object-classification stack (same reason zml/benchmarks/registry.py imports lazily).
        from zml.eval.imagenet_eval import Config as ImageNetConfig, main as imagenet_main

        run_eval(params, args.config, args.output_dir, experiment_name,
                 config_cls=ImageNetConfig, eval_fn=imagenet_main)
    elif mode == "face":
        # Imported here so the fire/nudity/imagenet eval paths keep no import-time dependency on the
        # ONNX face stack (same reason zml/benchmarks/registry.py imports lazily).
        from zml.eval.face_eval import Config as FaceConfig, main as face_main

        run_eval(params, args.config, args.output_dir, experiment_name,
                 config_cls=FaceConfig, eval_fn=face_main)
    else:
        raise ValueError(f"Unknown mode '{mode}'; expected 'eval', 'imagenet', 'face' or 'generate'.")
