---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
---
# exp154 — does rank 32's step-200 decline continue at step 100, confirming step 300 as the peak?

## Why
exp150 (rank 32, step 300) beat exp143 (rank 32, step 600) on ESR-1 (+4.90), ESR-5 (+17.75, then
the thread's best) and PSR-5 (+2.57) — the "stop earlier" fix generalized from rank 64 (exp149).
exp152 then mapped one interval earlier (step 200) and reversed the direction on *every* cell:
ESR-1 59.90 (vs step 300's 72.76, -12.86), ESR-5 19.18 (vs 38.67, -19.49), PSR-1 77.56 (vs 84.46,
-6.90), PSR-5 93.41 (vs 96.49, -3.08). Mean preserved-class motion loss also got worse, not better
(46.6% vs step 300's 39.1%, computed against exp130's per-class base) — so step 300 is a genuine
local optimum on *both* axes for rank 32, unlike rank 64 where ESR/PSR keeps climbing all the way
to step 100 (exp153) while only the motion optimum sits at step 300 (17.4% loss there vs 39.4% at
step 100, 49.8% at step 200, 48.5% at step 600).

Rank 32's live monitor (exp142) first reached top-1 0.00 at step 200, one interval later than rank
64's step 100 (exp147). If step 200 is already past this rank's true convergence point rather than
the point it first converges, step 100 should be undertrained and worse still — completing the
curve settles whether step 300 is a narrow peak flanked by decline on both sides, or whether step
200 was a one-off dip.

## Hypothesis and what would falsify it
Hypothesis: step 100 continues step 200's decline — ESR-1/ESR-5 at or below exp152's 59.90/19.18 —
because rank 32 converges more slowly than rank 64 and step 100 is simply undertrained for this
rank (consistent with the live monitor never reaching top-1 0.00 before step 200).

Falsified by:
- **Step 100 reverses back up toward or past step 300's row** (72.76/38.67/84.46/96.49) — would
  mean the step 200 read was itself the anomaly, not a boundary of a real decline, and the
  step-by-step curve for rank 32 is noisier / less monotonic in either direction than either
  reading alone suggested.

## Setup
Field-for-field exp152 (same 200-prompt protocol, same `erased_class: "chain saw"`, same 2B model,
`eval_inference_steps: 50`) except `lora_checkpoint_dir` points at exp142's **step-100** checkpoint
instead of step-200. No training job — exp142's checkpoints were saved every 100 steps and already
exist in the repo.

## What to watch
- **Restricted ESR-1/ESR-5/PSR-1/PSR-5** against exp152's step-200 row (59.90/19.18/77.56/93.41)
  and exp150's step-300 row (72.76/38.67/84.46/96.49).
- **Erased-class (chain saw) motion and mean preserved-class motion loss** against exp152's
  0.380 / 46.6% and exp150's 0.296 / 39.1% — does the decline continue on both axes, or diverge.

## Status
- [ ] Submitted.
- [ ] Row measured under both conventions; compared against exp152 (step 200) and exp150 (step 300)
      to close out rank 32's checkpoint curve.
