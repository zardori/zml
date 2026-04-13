from argparse import ArgumentParser
from pathlib import Path

import mlflow
import yaml

from zml.unlearn.unlearn_model import Config, main


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory for outputs and checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        params = yaml.safe_load(f)

    # Group runs by experiment folder name (e.g. exp002_esd_fire_lora8)
    experiment_name = Path(args.config).parent.name
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_artifact(args.config)
        config = Config(**params, output_dir=args.output_dir)
        main(config)
