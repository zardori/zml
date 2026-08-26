---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  FIRST FALSIFIER FIRED: step 200 is worse than step 300 on every ESR/PSR cell, confirming step
  300 is a genuine local peak for rank 32, not a point on a monotonic "earlier is better" trend.
  Restricted (10-way) row: ESR-1 59.90, ESR-5 19.18, PSR-1 77.56, PSR-5 93.41. Against exp150's
  step-300 read of the SAME rank-32 LoRA (72.76 / 38.67 / 84.46 / 96.49): ESR-1 down 12.86, ESR-5
  down 19.49 (nearly half), PSR-1 down 6.90, PSR-5 down 3.08 — every cell worse, unlike rank 64
  where the classification curve kept improving from step 300 through step 100 (exp149→exp151→
  exp153). Mean preserved-class motion loss (vs exp130's per-class base) is also worse here (46.6%)
  than at step 300 (39.1%, recomputed exactly from the quality block) or step 600 (44.0%, exp143)
  — so step 300 is a local optimum on BOTH axes for rank 32, not just the classification one. Chain
  saw's own motion is 0.380, well clear of the 0.15 guard floor. Rank 32's live monitor (exp142)
  first reached top-1 0.00 at step 200, one interval later than rank 64's step 100 (exp147) — this
  is consistent with step 200 being close to rank 32's actual convergence floor rather than a
  comfortably-converged early checkpoint, unlike rank 64 where step 100 was already well past
  convergence. exp154 checks step 100 to see whether the decline continues (rank 32 undertrained
  below step 300) or step 200 was itself the anomaly.
---
# exp152 — does rank 32's ESR/PSR curve keep rising at step 200, and does its motion margin erode
# the way rank 64's did?

## Why
exp150 found exp142's rank-32 checkpoint clearly better at step 300 than step 600 on ESR-1
(+4.90), ESR-5 (+17.75, now the thread's best), and PSR-5 (+2.57) — the "stop earlier" fix from
exp149 generalizes across capacity. But its SECOND finding did not generalize: rank 64's step-300
checkpoint cut preserved-class motion loss to a third of its step-600 read (~17% vs ~48%); rank
32's step-300 checkpoint costs slightly MORE preserved motion than step 600 (39.1% vs 35.9%), not
less. exp151 then mapped rank 64 one point earlier (step 200) and found the ESR/PSR curve still
rising (every cell better than step 300) while the motion picture got worse, not better — margin
over the 0.15 floor shrank to 0.026, and preserved-class motion loss rose to 49.8%, the worst
reading of any checkpoint measured. So for rank 64, step 300 looks like a genuine local motion
optimum on a curve where ESR/PSR keeps climbing, not a point on a monotonic motion trend either.

Rank 32 has one fewer data point than rank 64 at this point in the sweep (step 600, step 300;
rank 64 has step 600, step 300, step 200). This fills in the missing point: does rank 32 show the
same "ESR/PSR keeps rising, motion doesn't" shape one interval earlier, or does its already-odd
motion behavior (worse at step 300, not better) mean it degrades differently?

## Hypothesis and what would falsify it
Hypothesis: step 200 improves ESR-1/ESR-5/PSR-5 over exp150's step-300 read (72.76 / 38.67 / 96.49)
the same direction exp151 found for rank 64, while preserved-class motion loss stays elevated or
worsens further (it was already worse at step 300 than step 600 for this rank, unlike rank 64).

Falsified by:
- **Step 200 worse than step 300 on ESR** — would mean rank 32's slower live-monitor convergence
  (top-1 first hits 0.00 at step 200, not rank 64's step 100) means step 200 is this rank's true
  floor for convergence, and the curve does not extend as far back as it does for rank 64.
- **Preserved-class motion loss drops sharply at step 200** (back toward rank 64's ~17% pattern)
  — would mean rank 32's step-300 motion regression was a one-off, not evidence the two ranks
  degrade differently, and undercut the "classification and motion preservation are dissociating"
  reading from exp150/exp151 taken together.

## Setup
Field-for-field exp150 (same 200-prompt protocol, same `erased_class: "chain saw"`, same 2B model,
`eval_inference_steps: 50`) except `lora_checkpoint_dir` points at exp142's **step-200**
checkpoint instead of step-300. No training job — exp142's checkpoints were saved every 100 steps
and already exist in the repo.

## What to watch
- **Restricted ESR-1/ESR-5/PSR-1/PSR-5** against exp150's step-300 row (72.76 / 38.67 / 84.46 /
  96.49) and exp143's step-600 row (67.86 / 20.92 / 85.28 / 93.92).
- **Erased-class (chain saw) motion and its margin over the 0.15 floor** against exp150's 0.296 —
  whether rank 32 shows exp151's thinning-margin pattern (0.378 -> 0.176 for rank 64) or holds up
  better.
- **Mean preserved-class motion loss vs exp130's per-class base** against exp150's 39.1% — does it
  rise further (extending rank 32's odd "earlier is worse for motion" pattern) or reverse.

## Status
- [x] Submitted. Completed on helios, job 21162147.
- [x] Row measured under both conventions; compared against exp150 (rank 32, step 300) and exp151
      (rank 64, step 200). Falsifier 1 fired (step 200 worse than step 300 on ESR), so this rank's
      curve does NOT extend as far back as rank 64's. Follow-up: exp154 (step 100).
