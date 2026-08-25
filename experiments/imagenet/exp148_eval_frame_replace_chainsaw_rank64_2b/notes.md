---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
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

## Status
- [ ] Submitted.
- [ ] Row measured under both conventions; checked against GOAL.md's target table and all four
      guards.
- [ ] Compared against exp143's rank-32 row to classify the capacity curve as continuing,
      plateauing, or reversing, and decide the next lever accordingly.
