---
status: active
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  Split-prompt frame_replace dataset for 'chain saw', 30 triples (seeds 3201-3230) — the easy,
  spatially localized half of the object pilot.
---
# exp066 — split-prompt frame_replace dataset for "chain saw"

## Goal
First frame_replace training dataset for an ImageNet object class. A chain saw on a workbench is
present for the whole clip, so — exactly as with nudity — the partiality has to be manufactured by
the split-prompt sampler (`docs/split_prompt.md`) before the frame-local edit has anything to work
with.

Per triple: generate the combined chain-saw / object-free clip (`x0_original`), classify every frame
with ResNet-50, lift to a latent-frame concept mask, `edit_latent` onto interpolated donor frames
(`x0_edited`), save both. 30 triples, seeds 3201-3230.

## Setup
Same knobs as exp061, with the detector chosen by `concept: object` + `concept_target: "chain saw"`
through `zml/benchmarks/registry.py`. `concept_region: random` and `split_jitter: 2` keep the object's
temporal position decorrelated from the edit, so the trainer cannot learn "copy the object-free half".

The B prompts swap the chain saw for an object **outside the ten protocol classes** (bicycle pump,
watering can, paint can, ...) and are varied across the file — substituting one of the nine preserved
classes would corrupt PSR, and always substituting the *same* object would teach a fixed replacement.

`./submit_job.py helios experiments/exp066_split_chainsaw_dataset/config.yaml`

**Depends on exp064:** `frame_concept_threshold: 0.15` is a starting guess. Calibrate it against
exp064's `chain_saw/` videos locally first — a classifier probability is not on the same scale as a
NudeNet detection score, and a wrong threshold silently produces an all-skip or an all-mask dataset.

## What to watch
- Keep rate in `metadata.json` vs `skipped.json` (`no_concept`, `insufficient_donor_frames`). exp061
  kept 20 of 29; a much lower rate points at the threshold.
- `videos/*_original.mp4` vs `*_edited.mp4` by hand. Two failure modes to look for: the splice never
  rendered the chain saw (should have been skipped), and the exp055 frozen-donor artefact where the
  edited region holds still instead of continuing the scene.
- `concept_region` first/second balance across kept entries.

## Downstream
Feeds exp069 — fill its `metadata_file` / `latents_dir` with this run's `outputs_{timestamp}`.

## Status
- [ ] Threshold calibrated from exp064.
- [ ] Submitted.
- [ ] Dataset reviewed; kept/skipped counts recorded.
