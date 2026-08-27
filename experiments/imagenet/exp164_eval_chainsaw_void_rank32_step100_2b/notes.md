---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  HYPOTHESIS FALSIFIED: step 100 does NOT continue rising toward step 200, it sits well below it.
  Restricted (10-way) row: ESR-1 62.76, ESR-5 16.94, PSR-1 88.27, PSR-5 96.79. Against exp163
  (same LoRA, step 200: 86.73 / 43.57 / 82.95 / 93.45): ESR-1 down 23.97, ESR-5 down 26.63 -- both
  below the falsifier bar, so this run does NOT mirror rank 64's monotonic climb all the way to
  step 100 (exp149->exp151->exp153). Instead the four checkpoints now measured on this LoRA (step
  600 exp161: 45.71/10.82, step 300 exp162: 60.41/18.47, step 200 exp163: 86.73/43.57, step 100
  here: 62.76/16.94) trace a sharp single-peak curve centered on step 200, the same non-monotonic
  shape rank 32 alone showed on the random-distractor dataset (peak at step 300: exp150; step 200
  below it: exp152; step 100 partial recovery: exp154) -- just with the peak shifted one interval
  earlier. PSR-1/PSR-5 are this LoRA's best of the four checkpoints, continuing the pattern where
  weaker erasure reads as stronger preservation (the mirror image of collateral damage, not an
  independent gain). Chain saw's own motion is 0.8204, the healthiest margin of any checkpoint on
  this LoRA (base 0.840, floor 0.15) -- consistent with early training doing little erasure damage
  yet. Rank 32's checkpoint sweep on the void-target dataset is now complete at 100-step
  resolution; exp165 maps steps 125-275 (interval 25) to locate the peak's edges and check whether
  a nearby step keeps most of step 200's ESR gain while clearing the motion floor step 200 itself
  missed.
---
# exp164 — eval: chain-saw void-target dataset x rank 32, CogVideoX-2B, step 100

## Why
Companion to exp163 (step 200) — see its notes.md for the full rationale. exp161 (step 600) and
exp162 (step 300) show void+rank32 underperforming both single levers on ESR-1/ESR-5, with a
partial recovery from step 600 to step 300. This is the earliest saved checkpoint from exp160's
run, testing the far end of the training-length trend in one tick alongside exp163 rather than
waiting for exp163's result first — the two evals are independent (same completed training run,
different saved checkpoints, no dependency between them).

## Hypothesis and what would falsify it
Hypothesis: step 100 continues or plateaus the step-600→300 trend, mirroring rank 64's monotonic
climb all the way to step 100 (exp149→exp151→exp153) rather than rank 32 alone's non-monotonic
curve (peak at step 300, exp150; decline at step 200, exp152; partial recovery at step 100,
exp154). exp160's live monitor already showed top-1 reaching 0.00 by step 100 (faster than rank
32's random-distractor run, which took until step 200) and step 100 has this run's healthiest live
motion read (0.712) — consistent with void-target data accelerating convergence the way exp157/158
found for rank 8.

Falsified by: this checkpoint scoring at or below exp163's step-200 read on ESR-1 or ESR-5 (the
trend reverses before reaching the earliest checkpoint, mirroring rank 32 alone's own non-monotonic
shape rather than rank 64's monotonic one).

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp160's `frame_replace_lora_step100` checkpoint,
identical 200-prompt protocol to every other row in this thread.

## Status
- [ ] Submitted.
- [ ] Compared against exp163 (same LoRA, step 200), exp154 (rank 32, random-distractor, step 100)
      and exp153 (rank 64, random-distractor, step 100, this thread's overall best row).
