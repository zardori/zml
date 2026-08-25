---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  REVERSES, BY BOTH OF ITS OWN PRE-REGISTERED CRITERIA AT ONCE. Restricted (10-way) row: ESR-1
  71.53, ESR-5 16.43, PSR-1 76.20, PSR-5 92.45. Against exp143's rank-32 row (67.86 / 20.92 /
  85.28 / 93.92): ESR-1 is up (+3.67) but ESR-5 is DOWN (-4.49, below exp143's read and only
  +0.82 over exp134's rank-8 baseline of 15.61 — almost the whole rank-32 gain on the metric that
  matters is gone) and PSR-1 drops 9.08 points (85.28 -> 76.20, though still well above the 54.03
  floor). Both of the pre-registered "reverses" conditions fire (ESR-5 below 20.92, and a PSR cell
  below exp143's), so per the falsifier this rank increase "overfits the small 47-row dataset or
  starts trading preservation for erasure the way eta did" — the capacity lever peaks at rank 32
  for this dataset, exactly as exp143 predicted it might. Chain saw's own restricted top-5 is
  0.8357, HIGHER (worse) than exp143's 0.7908, confirming the ESR-5 regression is not noise — rank
  64 is measurably worse than rank 32 at suppressing the object from the top-5 guess, not just
  flat. The erased-class motion guard still passes (0.1814 vs the 0.15 floor) but the margin keeps
  shrinking with every capacity increase — exp134 (rank 8) 0.390 -> exp143 (rank 32) 0.223 ->
  here 0.181 — now within 0.03 of the floor. Preserved-class motion loss (vs exp130's per-class
  base) is also the worst yet: mean ~48% across the nine non-chain-saw classes (cassette player
  worst at -93%, matching exp137/exp140's recurring weak spot almost exactly; tench the only class
  essentially unaffected at +0.3%), against exp134's ~32% / exp137's ~36% / exp140's ~39% —
  capacity's collateral cost is growing monotonically with rank even though ESR-5 is not. Net: the
  capacity curve is non-monotonic — rank 8 -> 32 bought +5.31 ESR-5 with no PSR cost, rank 32 -> 64
  gives most of that back (-4.49 ESR-5) while costing 9 points of PSR-1 and the largest motion hit
  in the thread. exp143's rank-32 checkpoint remains this thread's best full-protocol row against
  GOAL.md's target. exp149 checks whether the regression is rank itself or this run's fixed
  600-step budget over-training a LoRA that (per exp147's live monitor) converges faster than
  every prior rank — an eval-only diagnostic on exp147's own earlier checkpoints, no new training.
---
# exp148 — reported ESR/PSR for exp147's rank-64 chain-saw LoRA (2B), the capacity lever's second doubling

## Why
exp143 found capacity (LoRA rank) is this thread's first non-null single-lever result: rank 8 -> 32
moved restricted ESR-1 49.90 -> 67.86 (+17.96) and ESR-5 15.61 -> 20.92 (+5.31), with both PSR
cells improving too — no trade-off, unlike eta (exp137, which bought ESR-1 at PSR's expense) or
dataset size (exp140, which moved nothing). Still, exp143 landed well short of GOAL.md's target
(ESR-1 92.38, ESR-5 77.09 — 56.17 points short on ESR-5, the binding guard), so exp147 doubled rank
again (32 -> 64) at the identical lr/step budget to test whether the gain continues.

exp147's live 9-prompt monitor converged cleanly and even faster than every prior rank (top-1 0.00
from step 100, versus rank 8/32's step ~150-200), which clears the standing "queue a full eval only
if the live monitor is healthy" gate this thread has used since exp142 -> exp143. But the live
top-5 signal is uninformative here: it matches exp142's already-near-zero band (0.00 at most
checkpoints) rather than dropping further, and exp139 (rank 8) -> exp142 (rank 32) already showed a
9-prompt sample can floor at 0 while the full protocol still finds a real several-point ESR-5
difference between ranks. So this eval is not confirming a lead the live sample raised — it is the
only instrument that can settle the question at all.

## Hypothesis and what would falsify it
Hypothesis: restricted ESR-5 continues to improve over exp143's rank-32 read of 20.92 by an amount
comparable to the rank 8->32 jump (+5.31), i.e. capacity is a lever with further room, not one that
saturated at rank 32.

Falsified by (three distinguishable outcomes, per exp143's own framing of the capacity question):
- **Continues at a similar rate** (ESR-5 comfortably above ~26, roughly extrapolating the 8->32
  slope) — capacity has more room; rank 128 or a still-larger jump becomes the next test.
- **Plateaus** (ESR-5 within a few points of 20.92, PSR unchanged or better) — capacity gains are
  front-loaded in the low ranks and diminish; would argue for combining capacity with a second lever
  (e.g. more steps at rank 32/64) rather than a further rank increase.
- **Reverses** (ESR-5 below 20.92, or a PSR cell drops below exp143's) — rank 64 overfits the small
  47-row dataset or starts trading preservation for erasure the way eta did; would cap the useful
  rank at 32 for this dataset size.

The erased-class motion guard (0.15 floor) is also checked: exp147's live sample ended at 0.089,
under floor, but exp142's identical-shape live-sample motion collapse (0.061) did not generalize —
exp143's full-protocol number came in at 0.223. Same read applies here unless the full number says
otherwise.

## Setup
Field-for-field exp143 except `lora_checkpoint_dir` points at exp147's final checkpoint
(`experiments/imagenet/exp147_frame_replace_chainsaw_rank64_2b/outputs_20260824_234931/frame_replace_lora_step600`).
Same 200-prompt protocol, same `erased_class: "chain saw"`, same `eval_inference_steps: 50`.

## What to watch
- **Restricted ESR-1/ESR-5/PSR-1/PSR-5** against GOAL.md's target table and all four guards, and
  against exp143's rank-32 row (67.86 / 20.92 / 85.28 / 93.92) to read the shape of the capacity
  curve at a second doubling.
- **Restricted top-5 on chain saw itself** — exp143's 0.7908 was the first read below the ~0.84-0.85
  band every rank-8 run was stuck at; whether rank 64 pushes it lower is the concrete number behind
  any further ESR-5 gain.
- **Erased-class motion** against exp143's 0.223 (already this thread's thinnest margin above the
  0.15 floor) — capacity increases have so far *reduced* the margin each time (exp134 0.390 ->
  exp143 0.223), so this is the guard most likely to bind next if the trend continues.

## Results (2026-08-25) — reverses on both pre-registered criteria

Completed on helios (job 21120698, 9426s of a 14h budget). Restricted (10-way) row:

| metric | exp134 (rank 8) | exp143 (rank 32) | exp148 (rank 64) | GOAL.md |
|---|---|---|---|---|
| ESR-1 | 49.90 | 67.86 | **71.53** | 92.38 |
| ESR-5 | 15.61 | 20.92 | **16.43** | 77.09 |
| PSR-1 | 82.71 | 85.28 | **76.20** | ≥54.03 |
| PSR-5 | 93.19 | 93.92 | **92.45** | ≥82.14 |

ESR-1 keeps climbing (+3.67 over rank 32) but ESR-5 gives back almost the entire rank-32 gain
(-4.49, landing only +0.82 over the rank-8 baseline) and PSR-1 drops 9.08 points. Both
pre-registered "reverses" conditions fire simultaneously. Chain saw's own restricted top-5 is
0.8357 — worse than exp143's 0.7908, so the ESR-5 regression is a real change in the erased
class's own residual signal, not noise elsewhere in the ranking. Erased-class motion guard passes
(0.1814 vs the 0.15 floor) but the margin has now shrunk on every capacity step: 0.390 (rank 8) ->
0.223 (rank 32) -> 0.181 (rank 64). Preserved-class motion loss (vs exp130's per-class base) is
the worst yet at a mean ~48% across the nine non-chain-saw classes (cassette player -93%, the same
class that was worst under exp137's high-eta arm and exp140's larger dataset; tench the one class
essentially untouched at +0.3%) — worse than exp134's ~32%, exp137's ~36%, exp140's ~39%.

## Status
- [x] Submitted (helios job 21120698, completed 2026-08-25T15:08).
- [x] Row measured under both conventions; checked against GOAL.md's target table and all four
      guards — all four numeric guards pass (PSR-1/PSR-5/motion), the target itself is not met.
- [x] Compared against exp143's rank-32 row: REVERSES (ESR-5 down, PSR-1 down). exp143's rank-32
      checkpoint remains this thread's best full-protocol row. exp149 tests whether the regression
      is rank itself or this run's fixed 600-step budget over-training a LoRA that converges
      faster than every prior rank (eval-only, on exp147's already-saved earlier checkpoints).
