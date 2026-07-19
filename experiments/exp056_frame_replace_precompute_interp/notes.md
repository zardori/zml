# exp056 — frame_replace precompute with motion-preserving donor interpolation

## Goal
Rebuild the frame_replace edited-target dataset with the exp055 **fix A**: `edit_latent` now linearly
interpolates in latent space between the nearest fire-free frame before and after each fire block,
instead of hard-copying a single frozen donor across it. The frozen copy taught the model to hold
still and collapsed motion globally (exp055: concept −84%, related −43%, unrelated −29%). Interpolation
keeps a smooth trajectory across the block, so the target no longer teaches "hold still."

## Setup
Identical to exp042 (same `prompts/cogvideox_partial_fire_curated.csv`, same seeds, same 50-step
generation, guidance 6.0, fire threshold 0.5, min_nofire_frames 2). The ONLY change is the code in
`zml/unlearn/frame_replace_ops.py::edit_latent`. So exp056 vs exp042 is a clean A/B on donor
construction — same clips, different fire-frame fill. Edge fire blocks (at clip start/end) still fall
back to a one-sided copy; `donor_map` now records the endpoint frame(s) used ([lo, hi] interpolated,
[d] one-sided).

## What to check
- Edited MP4s (`save_videos: true`) should move smoothly across the former fire segments rather than
  freezing. A quick `zml/eval/motion.py` pass on the edited videos vs exp042's edited videos should
  show higher motion in the patched region.
- `skipped.json` counts should match exp042 (edit logic changed, skip logic did not).

## Downstream
Training run **exp057** consumes this dataset (metadata_file / latents_dir), same eta=2 / step-300
regime as exp053 run_002; then re-run the exp055 motion eval on the new checkpoint to confirm the
global motion collapse is fixed.

## Status
- [x] Submitted.
- [x] Results pulled — `outputs_20260718_170407`: 66 targets (matches exp042), 132 latents
  (edited + original), `donor_map` now list-valued. exp057 path filled.
