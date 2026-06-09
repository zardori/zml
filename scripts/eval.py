import contextlib
from argparse import ArgumentParser
from pathlib import Path

import mlflow
import wandb
import yaml

from zml.eval.eval_model import Config, main


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

    # `disable_mlflow` is read (not popped) so `Config` still receives it via **params.
    disable_mlflow = params.get("disable_mlflow", False)

    if not disable_mlflow:
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(experiment_name)

    # eval_model.main() routes through zml.unlearn.eval.evaluate(), which logs all eval
    # metrics to mlflow/wandb itself; the entrypoint only owns the run lifecycle.
    with (contextlib.nullcontext() if disable_mlflow else mlflow.start_run()):
        if not disable_mlflow:
            mlflow.log_params(params)
            mlflow.log_artifact(args.config)
        try:
            wandb.init(
                project="zml",
                entity="zardori-zml",
                name=experiment_name,
                config=params,
            )
            wandb.save(args.config)
        except Exception as e:
            print(f"WARNING: wandb init failed ({e}), continuing without W&B tracking.")
            wandb.init(mode="disabled")
        config = Config(**params, output_dir=args.output_dir)
        main(config)
        wandb.finish()
