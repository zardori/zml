#!/bin/bash

REPO_DIR="/net/pr2/projects/plgrid/plggtriplane/poblos/zml"
CHECKPOINTS_DIR="$REPO_DIR/checkpoints"

# Find all checkpoint directories (excludes .zip files)
for checkpoint in $(find "$CHECKPOINTS_DIR" -mindepth 2 -maxdepth 2 -type d | sort); do
    checkpoint_name=$(basename "$checkpoint")
    ngs_variant=$(basename "$(dirname "$checkpoint")")

    sbatch \
        --job-name="gen_${ngs_variant}_${checkpoint_name}" \
        --export=ALL,MODEL_CHECKPOINT="$checkpoint",NGS_VARIANT="$ngs_variant" \
        "$REPO_DIR/athena_slurms/tuned_gen_cog.sh"

    echo "Submitted: $ngs_variant / $checkpoint_name"
done