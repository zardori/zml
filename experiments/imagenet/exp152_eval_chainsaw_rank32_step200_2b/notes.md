---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
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
- [ ] Submitted.
- [ ] Row measured under both conventions; compared against exp150 (rank 32, step 300) and exp151
      (rank 64, step 200) to decide whether the ESR/PSR-vs-motion dissociation is a general
      property of short training or specific to rank 64.
