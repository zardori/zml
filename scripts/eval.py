import contextlib
from argparse import ArgumentParser
from pathlib import Path

import mlflow
import wandb
import yaml

from zml.eval.eval_model import Config, main as eval_main
from zml.eval.generate_videos import GenerateConfig, main as generate_main


def run_eval(params: dict, config_path: str, output_dir: str, experiment_name: str) -> None:
    # `disable_mlflow` is read (not popped) so `Config` still receives it via **params.
    disable_mlflow = params.get("disable_mlflow", False)

    if not disable_mlflow:
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(experiment_name)

    # eval_main() routes through zml.unlearn.eval.evaluate(), which logs all eval metrics to
    # mlflow/wandb itself; the entrypoint only owns the run lifecycle.
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
        eval_main(Config(**params, output_dir=output_dir))
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
        params = yaml.safe_load(f)

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
    else:
        raise ValueError(f"Unknown mode '{mode}'; expected 'eval' or 'generate'.")
