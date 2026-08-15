---
status: ready
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  frame_replace erasure of 'chain saw' — does the method erase an ImageNet object class,
  semantically rather than positionally? Unblocked by exp117; trains on 21 screened rows merged from
  exp117 (14 closeup) and exp066 run 2 (7 wide), the wide ones deliberately kept as the only
  counterweight to the closeup prompt template. Needs merge_dataset.sh run on the cluster first.
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

- Dataset: exp117 + exp066 (split-prompt manufactured partial clips), screened and merged — below.
- Retention: exp068's ten-class anchors minus chain saw (`retention_exclude`).
- Regime: `erase_input_latent: original`, velocity loss, `erase_esd_eta: 2`, t in [400, 1000),
  constant LR 5e-4, 600 steps, rank-8 LoRA, `gradient_accumulation_steps: 4`.

## Dataset: 21 rows, merged from two builds

exp117 unblocked this. Its object-dominant prompts took chain-saw yield from 7/30 to 14/30, and the
config now trains on those 14 plus exp066 run 2's 7 screened survivors.

Merging rather than taking exp117 alone is a deliberate call, on two grounds:

- **Size.** exp062, the nudity run whose recipe this copies field-for-field, trained on 31. Fourteen
  is thin for 600 steps at rank 8.
- **Framing diversity, which is the more important one.** All 30 exp117 prompts share the closeup
  scaffold — "in close view, filling much of the frame", static camera. The 20 chain-saw eval prompts
  are ordinary scenes. A LoRA trained only on frame-filling objects has to generalise across framing
  to score on the eval set, and exp066's rows are the only wide-framing clips available.

The two sources also happen to balance the positional shortcut: 6 first / 8 second and 4 first /
3 second merge to 10 / 11.

**Before submitting**, build the merged dataset on the cluster — `combined_dataset/` is gitignored
and the `.pt` latents only exist there:

```
./merge_dataset.sh --cluster helios \
  --output experiments/imagenet/exp069_frame_replace_chainsaw/combined_dataset \
  --source experiments/imagenet/exp117_split_chainsaw_closeup/outputs_20260815_014333_screened.json \
           experiments/imagenet/exp117_split_chainsaw_closeup/outputs_20260815_014333/latents \
  --source experiments/imagenet/exp066_split_chainsaw_dataset/outputs_20260808_235138_screened.json \
           experiments/imagenet/exp066_split_chainsaw_dataset/outputs_20260808_235138/latents
```

Then `./submit_job.py helios experiments/imagenet/exp069_frame_replace_chainsaw/config.yaml`.

## What to watch
Live eval writes `summary.json` every `save_interval`; read that first.
- **Erasure:** `concept_detection_rate` on the concept set should fall well below exp064's base level
  for chain saw.
- **Shortcut test:** the concept prompts are ordinary full chain-saw scenes with no object-free half.
  A drop there means the LoRA learned to remove the object, not to copy the clean half of a training
  clip.
- **Collateral:** the unrelated set (one prompt per preserved class) should hold its detection rate,
  clip score and motion near base. A collapse there is the PSR failure exp071 would confirm.
- Watch for overfitting: 21 rows against 600 steps. If the erase loss floors early while the eval
  barely moves, that is memorisation of 21 clips, and the answer is exp121's rows, not more steps.
- **Framing generalisation.** Two thirds of the training rows are frame-filling closeups; every eval
  prompt is an ordinary scene. If the concept set barely moves while training looks healthy, check a
  few eval videos before concluding the method failed — erasing only at closeup framing is a
  different (and more informative) failure than not erasing.

## Downstream
exp071 runs the full 200-video ESR/PSR eval on the resulting checkpoint. The live numbers here are a
progress signal, not the reported metric.

## Status
- [x] Datasets complete; config wired to exp117 + exp066 screened sets and exp068's anchors.
- [ ] `merge_dataset.sh` run on the target cluster.
- [ ] Submitted.
- [ ] Checkpoint chosen for exp071; results written up.
