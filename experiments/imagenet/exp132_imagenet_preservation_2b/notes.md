---
status: done
concept: imagenet
method: preservation/precompute
thread: imagenet
takeaway: >
  Clean rebuild, nothing to diagnose. `metadata.json` has exactly 30 entries (10 classes x 3
  prompts, matching exp068's shape) and every entry carries `scaling_factor: 1.15258426` — the
  2B VAE value, confirming the swap from 5b's 0.7 actually took and the assert in
  `unlearn_frame_replace.py` will pass. Generation-only (no detection/screening, same as exp068),
  so nothing here is a research finding beyond "the anchors exist and are keyed to the right
  VAE" — but that was the blocker exp131 flagged for starting 2B `frame_replace` training.
  Unblocks exp133.
---
# exp132 — preservation anchors for the ImageNet object protocol, on CogVideoX-2B

## Why
GOAL.md moves the object thread's base model to CogVideoX-2B. exp131 cleared the split-prompt
dataset gate on 2B (25/30 usable, trajectory mode, no code change needed), which unblocks a 2B
chain-saw `frame_replace` training run — exp069's role, ported to 2B. But that training run also
needs a retention (preservation) dataset, and exp068's anchors were generated on **5b**.

`zml/unlearn/unlearn_frame_replace.py:323-326` and `:346-349` hard-assert the erase and retention
metadata's `scaling_factor` against `pipe.vae.config.scaling_factor` for the model actually being
trained. Checked the two models' HF configs directly (`vae/config.json`): CogVideoX-2b's
`scaling_factor` is 1.15258426, CogVideoX-5b's is 0.7 — different VAE calibrations, so exp068's
latents would fail that assert the instant a 2B training config pointed at them. This is not a
theoretical risk; the code already guards it and would abort the job at import time, but there is no
point spending a job slot finding that out when the mismatch is checkable from the two config files.
A 2B run needs its own preservation build.

## Setup
Field-for-field exp068 (`experiments/imagenet/exp068_imagenet_preservation/config.yaml`), only
`model_id` changed to `THUDM/CogVideoX-2b`. Same `prompts/imagenet_preservation.csv` (10 classes x 3
prompts, seeds 4301-4330, disjoint from `prompts/imagenet_objects.csv` on purpose), same
generation-only recipe (`save_videos: false`, no detection or editing — this is anchor latents only).

## What to watch
- 30 entries in the resulting `metadata.json`, each with a `class_name`, mirroring exp068.
- `scaling_factor` field on each entry should read 1.15258426 (2B), not 0.7 (5b) — the value
  `unlearn_frame_replace.py` will assert against.
- Spot-check a few videos (if `save_videos` is flipped on for a rebuild) or the detector confidences
  if this ever gets a screening pass — exp068 itself only checked that the base model renders each
  class, not per-clip quality, since this is a generation-only anchor set.

## Downstream
Feeds the 2B chain-saw `frame_replace` training run (`retention_metadata_file` /
`retention_latents_dir`), same role exp068 plays for the 5b thread, and every later 2B class in the
protocol without a rebuild.

## Status
- [x] Submitted.
- [x] `metadata.json` has 30 entries with `scaling_factor: 1.15258426`.
