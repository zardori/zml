"""Thin entrypoint for search jobs: parse the config and run the partial-fire prompt search.

Writes its outputs (results.jsonl, accepted_pairs.csv, summary.json, proposer_log.jsonl, videos/)
into the per-run `outputs_{timestamp}` dir (`--output_dir`). No mlflow/wandb run is opened here.
"""

from argparse import ArgumentParser

import yaml

from zml.search.partial_fire_search import SearchConfig, main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory for outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        params = yaml.safe_load(f)

    params.pop("slurm_time", None)  # infra key, not a search param
    params.pop("job_type", None)  # infra key, selects the entrypoint; not a search param

    main(SearchConfig(**params, output_dir=args.output_dir))
