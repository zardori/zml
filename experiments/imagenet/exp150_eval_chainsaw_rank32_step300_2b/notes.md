---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
---
# exp150 — does rank 32 also overtrain by step 600, the way exp149 showed rank 64 does?

## Why
exp149 found exp147's rank-64 checkpoint is measurably WORSE at step 600 (exp148's read) than at
step 300 (half the training budget) — not just flat. ESR-1 74.49 -> would-be-lower, ESR-5 21.63
vs exp148's 16.43, chain-saw top-5 0.7837 vs 0.8357, erased-class motion 0.378 vs 0.181, and mean
preserved-class motion loss ~17% vs ~48%. exp147's own live monitor showed concept top-1 already
at 0.00 by step 100, so the 500 remaining steps of exp148's run were training against an
already-erased target, with the extra 300 steps between exp149 and exp148 buying nothing but
regression on every axis measured.

This thread's rank-32 checkpoint (exp142/exp143) converged more slowly than rank 64 on its live
monitor (top-1 0.00 from step 200, not step 100) but by the same logic may still have 300+ steps of
margin between "erased" and "stopped," and exp143's row is currently tied with exp149 for this
thread's best (exp143 wins PSR by a few points, exp149 wins ESR by more). If rank 32 shows the same
overtraining shape, the fix is not "pick the best rank" but "stop earlier at every rank" — a
recipe change with more leverage than any single checkpoint choice, and one that would need
revisiting for every future capacity/eta/dataset run in this thread, not just this dataset.

## Hypothesis and what would falsify it
Hypothesis: rank 32's step-300 checkpoint matches or beats exp143's step-600 read on ESR-1/ESR-5
while giving up little or no PSR, mirroring exp149's shape one rank down — evidence that "stop
before the fixed 600-step budget" is a general recipe fix, not a rank-64-specific accident of its
faster convergence.

Falsified by:
- **Step 300 clearly worse than step 600 on ESR** (undertrained) — would mean rank 32's slower
  live-monitor convergence (step 200 vs rank 64's step 100) means it genuinely needs closer to the
  full 600 steps, and exp149's finding is specific to how fast a given rank converges, not a
  property of the 600-step budget itself.
- **Step 300 matches step 600 on every metric** (flat) — would mean rank 32 was already stable by
  step 300 with nothing to gain OR lose from stopping there, unlike rank 64's measurable
  regression — informative either way but would not generalize the "shorter budget helps"
  prescription.

## Setup
Field-for-field exp143 (same 200-prompt protocol, same `erased_class: "chain saw"`, same 2B model,
`eval_inference_steps: 50`) except `lora_checkpoint_dir` points at exp142's **step-300** checkpoint
instead of the final step-600 one. No training job — exp142's checkpoints were saved every 100
steps and already exist in the repo.

## What to watch
- **Restricted ESR-1/ESR-5/PSR-1/PSR-5** against exp143's step-600 row (67.86 / 20.92 / 85.28 /
  93.92) and exp149's rank-64/step-300 row (74.49 / 21.63 / 79.97 / 91.87).
- **Restricted top-5 on chain saw itself** (exp143: 0.7908) and erased-class motion (exp143: 0.223)
  — whether rank 32 shows the same "step 300 clears more margin above the 0.15 floor" pattern
  exp149 found (0.378 vs exp148's 0.181).
- **Mean preserved-class motion loss vs exp130's per-class base** — exp149's step-300 checkpoint
  more than halved this relative to its own step-600 read (~17% vs ~48%); whether rank 32 shows a
  comparable gap between its two stopping points.

## Status
- [ ] Submitted.
- [ ] Row measured under both conventions; compared against exp143 (rank 32, step 600) and exp149
      (rank 64, step 300) to decide whether "stop earlier than 600 steps" is a general recipe fix
      for this thread or specific to rank 64's faster convergence.
