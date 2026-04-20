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

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_artifact(args.config)
        wandb.init(
            project="zml",
            entity="zardori-zml",
            name=experiment_name,
            config=params,
        )
        wandb.save(args.config)
        config = Config(**params, output_dir=args.output_dir)
        metrics = main(config)

        for set_name, scores in metrics.items():
            mlflow.log_metric(f"eval/{set_name}_fire_detection_rate", scores["fire_detection_rate"])
            mlflow.log_metric(f"eval/{set_name}_clip_score_mean", scores["clip_score_mean"])
            mlflow.log_metric(f"eval/{set_name}_dover_technical_mean", scores["dover_technical_mean"])
            mlflow.log_metric(f"eval/{set_name}_dover_aesthetic_mean", scores["dover_aesthetic_mean"])

        wandb.log(
            {
                f"eval/{set_name}_{k}": v
                for set_name, scores in metrics.items()
                for k, v in [
                    ("fire_detection_rate", scores["fire_detection_rate"]),
                    ("clip_score_mean", scores["clip_score_mean"]),
                    ("dover_technical_mean", scores["dover_technical_mean"]),
                    ("dover_aesthetic_mean", scores["dover_aesthetic_mean"]),
                ]
            }
        )
        wandb.finish()
