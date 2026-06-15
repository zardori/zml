"""Thin entrypoint for precompute jobs: parse the config and dispatch to a precompute method.

Precompute writes its reusable dataset into the per-run `outputs_{timestamp}` dir (`--output_dir`),
just like the training/eval entrypoints; a training run that consumes it points at that directory.
No mlflow/wandb run is opened here.
"""

from argparse import ArgumentParser

import yaml


METHODS = {
    "frame_replace": "zml.precompute.frame_replace_precompute",
}


def _load_method(method: str):
    import importlib
    module = importlib.import_module(METHODS[method])
    return module.Config, module.main


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory for logs (unused by precompute)")
    args = parser.parse_args()

    with open(args.config) as f:
        params = yaml.safe_load(f)

    method = params.pop("method", "frame_replace")
    params.pop("job_type", None)  # infra key, not a precompute param
    params.pop("slurm_time", None)
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Valid options: {list(METHODS)}")

    Config, main = _load_method(method)
    main(Config(**params, output_dir=args.output_dir))
