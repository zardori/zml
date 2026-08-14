"""Rebuild a frame_replace dataset's *edited* latents from its saved originals — no GPU, no regen.

Why this exists
---------------
`frame_replace_split_precompute` saves both halves of every training pair: the manufactured
partial-concept clip (``original_latent_path``) and the concept-removed edit derived from it
(``latent_path``). Only the second depends on the edit rule. So when the edit rule changes, the
dataset can be *re-edited* from the originals instead of regenerated — which matters because
generation is the expensive part and, more importantly, because the source clips are unchanged, so
a human review that approved them still holds.

The case this was written for: exp061's triples were built 2026-08-02, three days before
``edit_latent_reflected`` replaced ``edit_latent``'s frozen single-frame fill. 20 of its 21
human-approved triples have a ``donor_map`` like ``[7, 7, 7, 7, 7]`` — one frame copied across the
whole concept block. Those 21 are 59% of exp080's training set, and exp080 collapsed concept motion
to -87%..-99% against a base of 0.686 while leaving unrelated motion untouched: the model learned
exactly what the targets encoded, "on a nudity prompt, emit a still image" (exp055 measured the same
pathology at -84%).

What it does per entry
----------------------
1. Load ``original_latent_path``.
2. Rebuild the masks from ``split_latent_frame``/``concept_region`` via the builder's own
   ``build_edit_masks``, applying the current ``boundary_margin``. The mask is knowable by
   construction, so this also upgrades datasets whose masks were detector-derived.
3. Apply ``edit_latent_reflected`` — mirror the safe segment's motion into the concept region.
4. Write the new edited latent, symlink the (unchanged) original, and emit updated metadata.

Detector fields are deliberately **dropped, not carried over**: ``edited_frame_confidences`` and
``edited_max_confidence`` describe the *old* edit and would be silently wrong against the new one.
Scoring them again needs a VAE decode, i.e. a GPU, which defeats the point — and the detector is
logging-only here (it gates nothing), so absence costs nothing. Fields describing the original clip
are unchanged and are carried through.

Usage (run where the latents live — normally on the cluster, like the merge step):

    uv run python tools/reedit_frame_replace_dataset.py \\
        --metadata experiments/nudity/exp061_split_nudity_dataset/metadata_human_filtered.json \\
        --latents-dir experiments/nudity/exp061_split_nudity_dataset/outputs_20260802_223148/latents \\
        --output-dir experiments/nudity/exp087_.../reedited
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from zml.precompute.frame_replace_split_precompute import (
    NUM_LATENT_FRAMES,
    build_edit_masks,
    edit_latent_reflected,
)

# Matches frame_replace_split_precompute's own default, so a re-edit lands where a fresh build would.
DEFAULT_BOUNDARY_MARGIN = 2

# Metadata keys that describe the *edit* and are therefore invalidated by re-editing.
STALE_EDIT_KEYS = ("edited_frame_confidences", "edited_max_confidence")


@dataclass
class ReeditStats:
    """What happened to one entry, for the run summary."""

    seed: int
    fills: int
    distinct_donors_before: int
    distinct_donors_after: int
    mask_changed: bool


def _distinct_donors(donor_map: dict | None) -> int:
    if not donor_map:
        return 0
    return len({v[0] if isinstance(v, list) else v for v in donor_map.values()})


def reedit_entry(
    entry: dict, latents_dir: Path, out_latents: Path, boundary_margin: int
) -> tuple[dict, ReeditStats]:
    """Re-edit one dataset entry; returns its new metadata and a before/after summary."""
    original_rel = entry["original_latent_path"]
    latent = torch.load(latents_dir / original_rel, map_location="cpu")

    split_frame, region = entry["split_latent_frame"], entry["concept_region"]
    concept_mask, edit_mask = build_edit_masks(split_frame, region, boundary_margin)
    edited, donor_map = edit_latent_reflected(latent, edit_mask, region)

    torch.save(edited, out_latents / entry["latent_path"])
    # The original is untouched, so link rather than copy it — the merge step does the same, and a
    # 100 MB tensor per triple is not worth duplicating.
    link = out_latents / original_rel
    if not link.exists():
        os.symlink(os.path.abspath(latents_dir / original_rel), link)

    new_entry = {k: v for k, v in entry.items() if k not in STALE_EDIT_KEYS}
    new_entry.update(
        concept_latent_mask=concept_mask,
        edited_latent_mask=edit_mask,
        boundary_margin=boundary_margin,
        donor_map={str(k): v for k, v in donor_map.items()},
        reedited_from=str(latents_dir),
    )
    stats = ReeditStats(
        seed=entry["seed"],
        fills=len(donor_map),
        distinct_donors_before=_distinct_donors(entry.get("donor_map")),
        distinct_donors_after=_distinct_donors(new_entry["donor_map"]),
        mask_changed=entry.get("concept_latent_mask") != concept_mask,
    )
    return new_entry, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True, help="Source metadata.json (or a human-filtered subset)")
    parser.add_argument("--latents-dir", required=True, help="Directory holding the source .pt files")
    parser.add_argument("--output-dir", required=True, help="New dataset root; latents/ is created inside")
    parser.add_argument("--boundary-margin", type=int, default=DEFAULT_BOUNDARY_MARGIN)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change, write nothing")
    args = parser.parse_args()

    latents_dir = Path(args.latents_dir)
    entries = json.load(open(args.metadata))
    out_root = Path(args.output_dir)
    out_latents = out_root / "latents"
    if not args.dry_run:
        out_latents.mkdir(parents=True, exist_ok=True)

    new_metadata, all_stats = [], []
    for entry in entries:
        if args.dry_run:
            concept_mask, edit_mask = build_edit_masks(
                entry["split_latent_frame"], entry["concept_region"], args.boundary_margin
            )
            all_stats.append(ReeditStats(
                seed=entry["seed"],
                fills=sum(edit_mask),
                distinct_donors_before=_distinct_donors(entry.get("donor_map")),
                distinct_donors_after=-1,
                mask_changed=entry.get("concept_latent_mask") != concept_mask,
            ))
            continue
        new_entry, stats = reedit_entry(entry, latents_dir, out_latents, args.boundary_margin)
        new_metadata.append(new_entry)
        all_stats.append(stats)

    if not args.dry_run:
        with open(out_root / "metadata.json", "w") as f:
            json.dump(new_metadata, f, indent=1)

    frozen_before = sum(1 for s in all_stats if s.distinct_donors_before == 1)
    frozen_after = sum(1 for s in all_stats if s.distinct_donors_after == 1)
    masks_fixed = sum(1 for s in all_stats if s.mask_changed)
    print(f"{'DRY RUN: ' if args.dry_run else ''}{len(all_stats)} entries from {args.metadata}")
    print(f"  frozen (single-donor) targets: {frozen_before} before"
          + ("" if args.dry_run else f" -> {frozen_after} after"))
    print(f"  concept masks corrected to construction: {masks_fixed}")
    if not args.dry_run:
        print(f"  wrote {out_root/'metadata.json'} and {len(all_stats)} edited latents to {out_latents}")


if __name__ == "__main__":
    main()
