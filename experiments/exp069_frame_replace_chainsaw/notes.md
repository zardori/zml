---
status: active
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  frame_replace erasure of 'chain saw' — does the method erase an ImageNet object class,
  semantically rather than positionally?
---
# exp069 — frame_replace erasure of "chain saw"

## Goal
The question the whole pilot exists to answer: does frame_replace erase an **ImageNet object** class,
and does it do so semantically rather than positionally? Chain saw is the easy half — a compact
object on a bench, which is the regime `docs/comparison_targets.md` §2.2 argues frame_replace was
designed for.

## Setup
Field-for-field identical to exp062 (nudity, eta=2) except the dataset, `concept`/`concept_target`,
and the retention set. Keeping the recipe fixed is the point: if chain saw erases and nudity did not,
the difference is the concept, not the hyperparameters.

- Dataset: exp066 (split-prompt manufactured partial clips).
- Retention: exp068's ten-class anchors minus chain saw (`retention_exclude`).
- Regime: `erase_input_latent: original`, velocity loss, `erase_esd_eta: 2`, t in [400, 1000),
  constant LR 5e-4, 600 steps, rank-8 LoRA, `gradient_accumulation_steps: 4`.

**Before submitting**, replace the `outputs_TIMESTAMP` placeholders with the real
`outputs_{timestamp}` directories from exp066 and exp068.

`./submit_job.py helios experiments/exp069_frame_replace_chainsaw/config.yaml`

## What to watch
Live eval writes `summary.json` every `save_interval`; read that first.
- **Erasure:** `concept_detection_rate` on the concept set should fall well below exp064's base level
  for chain saw.
- **Shortcut test:** the concept prompts are ordinary full chain-saw scenes with no object-free half.
  A drop there means the LoRA learned to remove the object, not to copy the clean half of a training
  clip.
- **Collateral:** the unrelated set (one prompt per preserved class) should hold its detection rate,
  clip score and motion near base. A collapse there is the PSR failure exp071 would confirm.
- Watch for overfitting: the dataset is at most 30 triples, likely fewer after review.

## Downstream
exp071 runs the full 200-video ESR/PSR eval on the resulting checkpoint. The live numbers here are a
progress signal, not the reported metric.

## Status
- [ ] exp066 and exp068 complete; timestamps filled in.
- [ ] Submitted.
- [ ] Checkpoint chosen for exp071; results written up.
