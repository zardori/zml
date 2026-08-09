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
``pull_results.sh --include-weights``) or directly on a cluster login node.

Run standalone, e.g.:
    uv run python -m zml.precompute.merge_frame_replace_datasets \\
        --source experiments/exp061_split_nudity_dataset/metadata_human_filtered.json \\
                 experiments/exp061_split_nudity_dataset/outputs_20260802_223148/latents \\
        --source experiments/exp078_.../grid_.../run_005/outputs/metadata_human_filtered.json \\
                 experiments/exp078_.../grid_.../run_005/outputs/latents \\
        --output_dir experiments/exp080_.../combined_dataset
"""

import argparse
import json
import os


def _relink_paths(d: dict, prefix: str, latents_dir: str, latents_out: str) -> None:
    """Symlink ``d``'s ``latent_path``/``original_latent_path`` (whichever are present) into
    ``latents_out`` under ``prefix``, rewriting ``d`` in place to point at the new names.

    Shared by the top-level entry and every ``variants[*]`` sub-dict, so a target variant's paths
    are relinked the same way its flat-key counterpart always has been.
    """
    for key in ("latent_path", "original_latent_path"):
        old_path = d.get(key)
        if old_path is None:
            continue
        new_name = prefix + os.path.basename(old_path)
        link_path = os.path.join(latents_out, new_name)
        if not os.path.exists(link_path):
            os.symlink(os.path.abspath(os.path.join(latents_dir, old_path)), link_path)
        d[key] = new_name


def merge(sources: list[tuple[str, str]], output_dir: str) -> None:
    latents_out = os.path.join(output_dir, "latents")
    os.makedirs(latents_out, exist_ok=True)

    combined: list[dict] = []
    for source_idx, (metadata_file, latents_dir) in enumerate(sources):
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

    print(f"Merged {len(sources)} sources -> {len(combined)} targets -> {output_dir}")


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
