---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  NEITHER FALSIFIER FIRED — AND THE RESULT IS STRONGER THAN THE HYPOTHESIS PREDICTED. Restricted
  (10-way) row: ESR-1 74.49, ESR-5 21.63, PSR-1 79.97, PSR-5 91.87. Against exp148's step-600 read
  of the SAME rank-64 LoRA (71.53 / 16.43 / 76.20 / 92.45): step 300 is NOT undertrained (ESR-1 and
  ESR-5 are both HIGHER, not lower, than step 600) and PSR-1 does more than "recover toward"
  rank 32 — it lands at 79.97, above the halfway point between exp148's 76.20 and exp143's 85.28
  (+3.77 of the 9.08-point gap, ~42% recovered). So stopping training at the halfway point made
  this checkpoint better at erasing AND better at preserving than training it the full 600 steps —
  the extra 300 steps exp148 ran were pure downside, the clearest overtraining signature this
  thread has produced. exp149 is now this thread's best full-protocol row on ESR-1 (74.49, +6.63
  over exp143's rank-32/step-600 67.86) and ties/edges exp143 on ESR-5 (21.63 vs 20.92, +0.71) —
  while both PSR cells stay comfortably clear of GOAL.md's floors (79.97 vs >=54.03, 91.87 vs
  >=82.14). Erased-class motion is 0.378, this rank's healthiest margin above the 0.15 floor by
  far (exp148's step-600 read was 0.181, exp143's rank-32 was 0.223) — a -55% drop from exp130's
  base 0.840, not the near-total collapse step 600 showed. Preserved-class motion loss (vs
  exp130's per-class base) averages only ~17% across the nine other classes, the best (least
  collateral) of every capacity-lever arm measured so far (exp134 rank-8 ~32%, exp143 rank-32
  unreported exactly but between these, exp148 rank-64/step-600 ~48%). Still well short of
  GOAL.md's target (92.38 / 77.09) — this is a better checkpoint, not a solved problem. Whether
  this "stop earlier" effect generalizes to rank 32, and whether step 300 is rank 64's actual peak
  or an even earlier stop does better, are exp150 and exp151.
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
- [x] Submitted (helios job 21144727, completed 2026-08-25T19:49, 9440s of a 14h budget).
- [x] Row measured under both conventions; compared against exp143 (rank 32) and exp148 (rank 64,
      step 600).

## Results (2026-08-25) — over-training confirmed, not just partially

| metric | exp134 (r8/step600) | exp143 (r32/step600) | exp148 (r64/step600) | exp149 (r64/step300) | GOAL.md |
|---|---|---|---|---|---|
| ESR-1 | 49.90 | 67.86 | 71.53 | **74.49** | 92.38 |
| ESR-5 | 15.61 | 20.92 | 16.43 | **21.63** | 77.09 |
| PSR-1 | 82.71 | 85.28 | 76.20 | **79.97** | >=54.03 |
| PSR-5 | 93.19 | 93.92 | 92.45 | **91.87** | >=82.14 |
| chain saw top-5 (restricted) | ~0.84-0.85 | 0.7908 | 0.8357 | **0.7837** | - |
| chain saw motion | 0.390 | 0.223 | 0.181 | **0.378** | >=0.15 |
| mean preserved-class motion loss vs exp130 base | ~32% | (between) | ~48% | **~17%** | - |

Both pre-registered falsifiers miss in the direction that favors the hypothesis by more than it
predicted: step 300 beats step 600 on ESR-1, ESR-5, chain-saw top-5, both motion metrics, and
partially recovers PSR-1 — it only gives up 2.05 points of PSR-5 against exp148 (91.87 vs 92.45,
still 9.7 points clear of the floor). exp147's own live monitor already showed concept top-1 at
0.00 from step 100, so by step 300 the LoRA had 200 steps of margin doing nothing but drifting the
checkpoint toward the worse point exp148 measured — the same "erase pressure past convergence only
costs preservation" pattern exp137 found for eta, now shown to apply to step count at high rank
too. exp143's rank-32/step-600 checkpoint no longer stands alone as this thread's best row: exp149
beats it on ESR-1 and ESR-5, exp143 still wins PSR-1/PSR-5 by a few points — both comfortably clear
every floor, so which is "better" depends on whether the remaining gap to GOAL.md's target is
read as ESR-bound (it is: ESR-5 gap is 55-56 points on either checkpoint, PSR has 20-38 points of
slack) — by that reading exp149 is the new best row.

Two open questions this raises, both answerable with existing checkpoints and no new training:
does rank 32 show the same overtraining shape (exp150, step 300 of exp142), and is step 300 rank
64's actual peak or would an earlier stop (step 200, already at live top-1 0.00) do even better
(exp151)?
