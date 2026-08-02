# exp068 — preservation anchors for the ImageNet object protocol

## Goal
PSR is half of the published comparison, so the retention branch should anchor the nine classes that
are actually being preserved. exp062 borrowed exp041's fire-era generic prompts, which was fine for a
"does erasure happen at all" pilot but would make PSR here uninterpretable — it would measure
preservation of classes the anchors never mention.

One dataset for all ten classes: 3 prompts per class, base-model `x0` saved unedited. Each erase run
sets `retention_exclude: <its class>` so the erased class's anchors are dropped — anchoring the
erased class would pull directly against the erase branch.

## Setup
`method: preservation`, 30 clips from `prompts/imagenet_preservation.csv`. That CSV carries a
`class_name` column, which `preservation_precompute` now copies into `metadata.json`; that column is
what `retention_exclude` filters on.

**The prompts are disjoint from `prompts/imagenet_objects.csv`.** This is deliberate and should stay
that way: anchoring on the exact 20 prompts PSR is measured on would be training on the test set.
The classes are what we preserve, not the specific items.

`./submit_job.py athena experiments/exp068_imagenet_preservation/config.yaml`

## What to watch
- 30 entries in `metadata.json`, each with a `class_name`.
- Spot-check that the base model actually renders each class in these prompts — an anchor whose clip
  does not contain its class anchors nothing useful.

## Downstream
Feeds exp069 and exp070 (`retention_metadata_file` / `retention_latents_dir`), and every later class
in the protocol without a rebuild.

## Status
- [ ] Submitted.
- [ ] Results pulled; `class_name` present in metadata; anchors spot-checked.
