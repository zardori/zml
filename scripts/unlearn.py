from argparse import ArgumentParser
from pathlib import Path

import mlflow
import wandb
import yaml

from zml.unlearn.unlearn_model import Config, main


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory for outputs and checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        params = yaml.safe_load(f)

    # Force filesystem backend so metrics/params land in mlruns/ and are rsync-able.
    # Without this, a MLFLOW_TRACKING_URI=sqlite://... in the environment (common on
    # shared clusters) would store metrics only in a local .db file that never gets synced.
    mlflow.set_tracking_uri("mlruns")

    # Group runs by experiment folder name (e.g. exp002_esd_fire_lora8)
    experiment_name = Path(args.config).parent.name
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_artifact(args.config)
        wandb.init(
            project="zml-unlearn",
            name=experiment_name,
            config=params,
        )
        wandb.save(args.config)
        config = Config(**params, output_dir=args.output_dir)
        main(config)
        wandb.finish()
