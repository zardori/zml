# exp054 — ESD + preservation + mask-based localization (fire)

## Goal
Add the T2VUnlearning localization regularizer `L_loc` (arXiv:2505.17550, Eq. 9) on top of the
proven `esd_preservation` recipe and check whether confining the LoRA edit to the concept region
(the "fire" token's text->video attention mask) improves the erasure/quality trade-off vs. exp008.

## Setup
- Same hyperparameters as exp008 (`esd_preservation`, rank 8, lr 5e-4, ng 1.0, 1000 steps).
- New knobs: `use_localization`, `localization_weight`, `mask_concept_word`, `mask_threshold`.
- Total loss = `loss_forget + preservation_weight*loss_preserve + localization_weight*loss_loc`.

## What to watch
- `train/loss_loc` in `summary.json` / `metrics.jsonl` — should trend down; forget/preserve losses
  should behave as in exp008.
- Control-set eval: `concept` fire score should drop while `related`/`unrelated` stay near base.
- Memory: localization disables gradient checkpointing on the student forget forward — prefer
  helios; if athena OOMs, cut `steps` for a first debug run.

## Results
_TBD_
