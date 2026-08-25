---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
---
# exp149 — does rank 64's regression come from over-training, or from capacity itself?

## Why
exp148 found the capacity lever reverses on its second doubling: rank 64's full-protocol row
(ESR-1 71.53, ESR-5 16.43, PSR-1 76.20, PSR-5 92.45) gives back almost all of rank 32's ESR-5 gain
(exp143: 20.92) and costs 9.08 points of PSR-1, while chain saw's own restricted top-5 (0.8357) is
*worse* than rank 32's (0.7908) — not just flat. Both of exp148's pre-registered "reverses"
conditions fired together.

But exp147's own live monitor showed rank 64 converging faster than every prior rank — concept
top-1 reaches 0.00 by step 100, versus rank 8/32's step ~150-200 — while every capacity run so far
has used the *same* fixed 600-step budget regardless of rank. That is a plausible confound: a
LoRA with more capacity that converges in a third of the steps may simply be overfitting the small
47-row dataset for the remaining two-thirds of training, the same way exp137 showed *more erase
pressure* (higher eta) buys nothing once the target is already reached and only costs preservation
from there. exp147's saved per-100-step checkpoints let this be tested with an eval job alone — no
new training.

## Hypothesis and what would falsify it
Hypothesis: rank 64's step-300 checkpoint (half the training budget) preserves ESR-5 close to the
step-600 read (16.43) while PSR-1 recovers toward rank 32's level (85.28) — i.e. the extra 300
steps are where preservation gets traded away for no further erasure gain, matching exp137's
"erase pressure past convergence only costs preservation" pattern.

Falsified by:
- **Step 300 undertrained** — ESR-1/ESR-5 measurably worse than step 600 (erasure not yet
  complete) with no better PSR-1. Would mean rank 64 needs the full 600 steps to erase at all, so
  the regression is not a stopping-point artifact — exp143's rank 32 stands as the capacity
  ceiling for this dataset, and the next lever is not step count.
- **Step 300 matches step 600 on every metric** (ESR and PSR both flat) — would mean the
  checkpoint is already saturated by step 300 either way, and the PSR-1 loss between step 300 and
  step 600 (if any) is not recoverable by this diagnostic, pointing back at rank/dataset-size
  interaction (exp148's "overfits the small 47-row dataset" reading) rather than at training
  length.

Either outcome is informative and answers a question exp148 raised but could not settle from a
single checkpoint.

## Setup
Field-for-field exp148 (same 200-prompt protocol, same `erased_class: "chain saw"`, same 2B model,
`eval_inference_steps: 50`) except `lora_checkpoint_dir` points at exp147's **step-300** checkpoint
instead of the final step-600 one. No training job — exp147's checkpoints were saved every 100
steps and already exist on the cluster where exp147 trained.

## What to watch
- **Restricted ESR-1/ESR-5/PSR-1/PSR-5** against exp148's step-600 row (71.53 / 16.43 / 76.20 /
  92.45) and exp143's rank-32 row (67.86 / 20.92 / 85.28 / 93.92).
- **Restricted top-5 on chain saw itself** (exp148: 0.8357, exp143: 0.7908) — whether the
  regression on the erased class's own residual signal is present at step 300 too.
- **Erased-class motion** against exp148's 0.181 (already the thread's thinnest margin above the
  0.15 floor) and preserved-class motion loss against exp130's per-class base (exp148: mean ~48%
  across the nine preserved classes, cassette player worst at -93%).

## Status
- [ ] Submitted.
- [ ] Row measured under both conventions; compared against exp143 (rank 32) and exp148 (rank 64,
      step 600) to decide whether the capacity lever's ceiling is rank-32-with-full-training or
      whether a shorter step budget at rank 64 recovers rank 32's result or better.
