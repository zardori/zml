"""Combine multiple frame_replace precompute outputs into one dataset directory.

``unlearn_frame_replace.py`` takes exactly one ``metadata_file``/``latents_dir`` pair for its edit
targets, so training on triples from more than one precompute run (e.g. exp061's original set plus
exp078's new close-up/multi-person triples) needs them merged into one directory first. This
symlinks each source's ``.pt`` files into a combined ``latents/`` dir (prefixed per source to avoid
filename collisions across runs) and writes one combined ``metadata.json`` with ``latent_path``/
``original_latent_path`` updated to match.

Also relinks every target's ``variants[*]`` block (``emit_whole_clip_target``,
``docs/face_identity.md``), with the same source prefix — otherwise a merged dataset's
``variants["wholeclip"]`` paths would still point at the unmerged source directory, and a
``target_variant: wholeclip`` training run would fail (or silently read from the wrong place) as
soon as its dataset came from a merge.

No GPU needed — this is pure file I/O, so it can run locally (after pulling the source latents with
``pull_results.sh --include-weights``) or directly on a cluster login node. ``merge_dataset.sh``
(repo root) is the local entrypoint that does the latter over ssh, since ``combined_dataset/`` is
gitignored and has to be built where it will be read from. It invokes this module with the login
node's own stock ``python3`` rather than ``uv run`` or the repo's ``.venv``: login nodes have no
``uv``, and the ``.venv`` (built by ``uv run`` on a compute node) has wheels for the GPU nodes'
architecture, which need not match the login node's (helios: aarch64 GH200 vs. x86_64 login node)
— it may not even be executable there. This module and ``zml.paths`` are stdlib-only and avoid the
PEP 585 ``list[...]``/``dict[...]`` annotation syntax for the same reason: helios' login node ships
Python 3.6, which predates it.

A source's ``metadata_file``/``latents_dir`` is resolved through ``zml.paths.resolve_input_path``,
the same peer-root fallback every training entrypoint uses — so this can merge sources that live in
a different project member's repo, as long as ``ZML_PEER_ROOTS`` is set (``slurm/peer_roots.sh``).

Run standalone, e.g.:
    python3 -m zml.precompute.merge_frame_replace_datasets \\
        --source experiments/nudity/exp061_split_nudity_dataset/metadata_human_filtered.json \\
                 experiments/nudity/exp061_split_nudity_dataset/outputs_20260802_223148/latents \\
        --source experiments/nudity/exp078_.../grid_.../run_005/outputs/metadata_human_filtered.json \\
                 experiments/nudity/exp078_.../grid_.../run_005/outputs/latents \\
        --output_dir experiments/nudity/exp080_.../combined_dataset
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

from zml.paths import resolve_input_path


def _relink_paths(d: dict, prefix: str, latents_dir: str, latents_out: str) -> None:
    """Symlink ``d``'s ``latent_path``/``original_latent_path`` (whichever are present) into
    ``latents_out`` under ``prefix``, rewriting ``d`` in place to point at the new names.

    Shared by the top-level entry and every ``variants[*]`` sub-dict, so a target variant's paths
    are relinked the same way its flat-key counterpart always has been.

    Raises if a source ``.pt`` doesn't exist, rather than writing a dangling symlink that would
    only fail hours later inside a training job.
    """
    for key in ("latent_path", "original_latent_path"):
        old_path = d.get(key)
        if old_path is None:
            continue
        target = os.path.abspath(os.path.join(latents_dir, old_path))
        if not os.path.exists(target):
            raise FileNotFoundError(f"Source latent missing, would create a dangling symlink: {target}")
        new_name = prefix + os.path.basename(old_path)
        link_path = os.path.join(latents_out, new_name)
        if not os.path.exists(link_path):
            os.symlink(target, link_path)
        d[key] = new_name


def merge(sources: List[Tuple[str, str]], output_dir: str) -> None:
    latents_out = os.path.join(output_dir, "latents")
    os.makedirs(latents_out, exist_ok=True)

    combined: List[Dict] = []
    for source_idx, (metadata_file, latents_dir) in enumerate(sources):
        metadata_file = resolve_input_path(metadata_file)
        latents_dir = resolve_input_path(latents_dir)
        with open(metadata_file) as f:
            entries = json.load(f)
        if not entries:
            raise ValueError(f"{metadata_file} has no entries.")
        prefix = f"src{source_idx}_"
        for entry in entries:
            new_entry = dict(entry)
            _relink_paths(new_entry, prefix, latents_dir, latents_out)
            if "variants" in new_entry:
                new_entry["variants"] = {
                    variant_name: dict(variant_dict)
                    for variant_name, variant_dict in new_entry["variants"].items()
                }
                for variant_dict in new_entry["variants"].values():
                    _relink_paths(variant_dict, prefix, latents_dir, latents_out)
            new_entry["_source_metadata_file"] = metadata_file  # provenance, not read by the trainer
            combined.append(new_entry)

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(combined, f, indent=2)

    n_links = sum(1 for _ in os.scandir(latents_out))
    print(f"Merged {len(sources)} sources -> {len(combined)} targets ({n_links} links) -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", nargs=2, action="append", metavar=("METADATA_FILE", "LATENTS_DIR"),
        required=True, help="Repeatable: one metadata.json + latents_dir pair per source dataset.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    merge([tuple(s) for s in args.source], args.output_dir)
