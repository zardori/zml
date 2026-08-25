---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
---
# exp151 — is step 300 rank 64's peak, or does an even earlier stop (step 200) do as well or better?

## Why
exp149 found exp147's rank-64 checkpoint measurably regresses between step 300 and step 600 on
every axis (ESR-1, ESR-5, chain-saw top-5, erased-class motion, preserved-class motion) — the 300
extra steps were pure downside. exp147's live 9-prompt monitor showed concept top-1 already at
0.00 by step 100, a full 200 steps before exp149's checkpoint. If step 300 already includes some of
the same overtraining exp148's step 600 showed in more severe form, step 200 — one save interval
earlier, still past the point live top-1 first hit 0.00 — could match or beat it. If step 300 is
already the peak (or step 200 is undertrained relative to it), that bounds how far back the
"stop earlier" fix can be pushed for this rank/dataset and gives the shape of the curve on the near
side of the optimum, not just the far side exp148/exp149 already mapped.

## Hypothesis and what would falsify it
Hypothesis: step 200 lands close to step 300's read (ESR-1 ~74, ESR-5 ~22, PSR-1 ~80) rather than
below it — i.e. the checkpoint was already near its full-protocol peak by step 200, consistent with
the live monitor's step-100 convergence, and the regression exp149 found is monotonic degradation
from a point at or before step 200, not a narrow peak centered on step 300.

Falsified by:
- **Step 200 clearly worse than step 300 on ESR** (undertrained on the full protocol despite the
  live monitor's step-100 top-1 read) — would mean the live 9-prompt signal at step 100-200
  overstates full-protocol convergence the way it has before (exp135, exp139), and step 300 is at
  or near the true near-side optimum, not an arbitrary earlier point on a still-rising curve.
- **Step 200 clearly better than step 300 on ESR with no PSR cost** — would mean the regression
  starts even earlier than step 300, and a still-shorter budget (e.g. step 100) is worth checking
  next.

## Setup
Field-for-field exp149 (same 200-prompt protocol, same `erased_class: "chain saw"`, same 2B model,
`eval_inference_steps: 50`) except `lora_checkpoint_dir` points at exp147's **step-200** checkpoint
instead of step-300. No training job — exp147's checkpoints were saved every 100 steps and already
exist in the repo.

## What to watch
- **Restricted ESR-1/ESR-5/PSR-1/PSR-5** against exp149's step-300 row (74.49 / 21.63 / 79.97 /
  91.87) and exp148's step-600 row (71.53 / 16.43 / 76.20 / 92.45) — where step 200 falls relative
  to both settles whether the curve is still rising, flat, or already past its peak at step 300.
- **Erased-class motion and mean preserved-class motion loss** against exp149's 0.378 / ~17% —
  whether an even earlier stop buys more motion margin or costs erasure to get it.

## Status
- [ ] Submitted.
- [ ] Row measured under both conventions; compared against exp149 (step 300) and exp148 (step 600)
      to map the near side of the training-length curve and decide the best stopping point for
      rank 64 on this dataset.
