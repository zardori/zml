#!/usr/bin/env -S uv run
"""Copy experiment inputs that exist on one cluster into another cluster's repo.

`submit_job.py` does this on its own for the config it is about to submit, so this script is for the
cases that are not a submission: staging the sources of a `merge_dataset.sh` build, or moving a
dataset ahead of time so the submission itself is instant. Paths are repo-relative, exactly as a
config writes them, and land at the same repo-relative path in your repo on the target cluster.

Usage:
    tools/sync_cluster_inputs.py <target-cluster> --config <config.yaml> [--from CLUSTER] [--yes]
    tools/sync_cluster_inputs.py <target-cluster> <repo-relative path>... [--from CLUSTER] [--yes]

Examples:
    # everything exp069 needs that helios does not have yet
    tools/sync_cluster_inputs.py helios --config experiments/imagenet/exp069_frame_replace_chainsaw/config.yaml

    # one directory, no config involved
    tools/sync_cluster_inputs.py helios experiments/imagenet/exp068_imagenet_preservation/outputs_20260803_233647/latents

Both sides search your repo and the peers' (slurm/check_config_paths.sh), so an input a teammate
produced counts as present. Transfers go cluster-to-cluster when the source login node can ssh to
the target (agent forwarding), otherwise they are streamed through this machine.
"""

import argparse
import sys

import yaml

from zml.cluster_sync import (
    ClusterSyncError,
    config_data_paths,
    fetch_missing_inputs,
    git_pull,
    load_cluster,
    locate_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy inputs that are missing on one cluster from another.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cluster", help="Target cluster: athena or helios")
    parser.add_argument("paths", nargs="*", help="Repo-relative paths to make available there")
    parser.add_argument(
        "--config",
        default=None,
        help="Take the paths from an experiment config instead of the command line",
    )
    parser.add_argument(
        "--from",
        dest="source",
        default=None,
        help="Source cluster to search (default: every other cluster, in order)",
    )
    parser.add_argument("--yes", action="store_true", help="Do not ask before copying")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = list(args.paths)
    if args.config:
        with open(args.config) as f:
            paths += [p for p in config_data_paths(yaml.safe_load(f)) if p not in paths]
    if not paths:
        print("Error: no paths given (pass them as arguments or via --config).", file=sys.stderr)
        sys.exit(1)

    try:
        target = load_cluster(args.cluster)
        git_pull(target, check=False)  # a path may simply not have been pulled there yet
        located = locate_paths(target, paths)
        for rel in located.found:
            print(f"  already on {target.name}: {rel}")
        if not located.missing:
            print(f"Nothing to do: all {len(paths)} path(s) are available on {target.name}.")
            return
        still_missing = fetch_missing_inputs(
            target,
            located.missing,
            sources=[args.source] if args.source else None,
            assume_yes=args.yes,
        )
    except ClusterSyncError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if still_missing:
        print(f"Still missing on {target.name} ({len(still_missing)}):", file=sys.stderr)
        for path in still_missing:
            print(f"  {path}", file=sys.stderr)
        sys.exit(1)
    print(f"Done: every path is now available on {target.name}.")


if __name__ == "__main__":
    main()
