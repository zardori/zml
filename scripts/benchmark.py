"""Thin entrypoint for benchmark/analysis jobs that score existing videos (no model, no training).

Dispatches on the config's ``method`` to a benchmark module, mirroring ``scripts/precompute.py``.
Used to run detectors over a directory of already-generated clips and write a report into the run's
``outputs_{timestamp}`` dir — the queued alternative to an interactive session.
"""

from argparse import ArgumentParser

import yaml

from zml.paths import resolve_config_paths


METHODS = {
    "nudity": "zml.benchmarks.nudity_report",
}


def _load_method(method: str):
    import importlib
    module = importlib.import_module(METHODS[method])
    return module.Config, module.main


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory for the benchmark report")
    args = parser.parse_args()

    with open(args.config) as f:
        params = resolve_config_paths(yaml.safe_load(f))

    method = params.pop("method", "nudity")
    params.pop("job_type", None)
    params.pop("slurm_time", None)
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Valid options: {list(METHODS)}")

    Config, main = _load_method(method)
    main(Config(**params, output_dir=args.output_dir))
