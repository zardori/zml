"""Combine multiple frame_replace precompute outputs into one dataset directory.

``unlearn_frame_replace.py`` takes exactly one ``metadata_file``/``latents_dir`` pair for its edit
targets, so training on triples from more than one precompute run (e.g. exp061's original set plus
exp078's new close-up/multi-person triples) needs them merged into one directory first. This
symlinks each source's ``.pt`` files into a combined ``latents/`` dir (prefixed per source to avoid
filename collisions across runs) and writes one combined ``metadata.json`` with ``latent_path``/
``original_latent_path`` updated to match.

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
            for key in ("latent_path", "original_latent_path"):
                old_path = entry.get(key)
                if old_path is None:
                    continue
                new_name = prefix + os.path.basename(old_path)
                link_path = os.path.join(latents_out, new_name)
                if not os.path.exists(link_path):
                    os.symlink(
                        os.path.abspath(os.path.join(latents_dir, old_path)), link_path
                    )
                new_entry[key] = new_name
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
