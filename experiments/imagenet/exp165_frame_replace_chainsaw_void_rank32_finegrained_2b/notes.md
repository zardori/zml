---
status: ready
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  Not yet run.
---
# exp165 — frame_replace: chain-saw void-target dataset x rank 32, CogVideoX-2B, fine-grained checkpoint sweep around the step-200 peak

## Why
exp160 trained exp156's void-target dataset at rank 32, saving checkpoints every 100 steps.
exp161/exp162/exp163/exp164 evaluated all four (600/300/200/100) on the full 200-prompt protocol
and found a single, sharp peak at step 200:

| step | ESR-1 | ESR-5 | PSR-1 | PSR-5 | chain-saw motion |
|---|---|---|---|---|---|
| 600 (exp161) | 45.71 | 10.82 | 90.25 | 97.07 | 0.546 |
| 300 (exp162) | 60.41 | 18.47 | 86.96 | 95.99 | 0.408 |
| 200 (exp163) | 86.73 | 43.57 | 82.95 | 93.45 | 0.138 |
| 100 (exp164) | 62.76 | 16.94 | 88.27 | 96.79 | 0.820 |

Step 200 is this thread's best classification row by a wide margin — ESR-1 within 5.65 points of
GOAL.md's 92.38 target, ESR-5 essentially tying the thread's previous best (exp153: 44.49). But its
motion (0.138) is the first fully-evaluated checkpoint in the whole rank/step sweep to actually
breach the 0.15 guard floor, where every other checkpoint on this same LoRA clears it by a wide
margin (0.408–0.820). The 100-step resolution used so far cannot tell whether that motion dip is a
narrow, single-checkpoint transient (in which case a nearby step might keep most of the ESR gain
while staying above 0.15) or the visible edge of a wider collapse this sweep is too coarse to
resolve.

## Hypothesis and what would falsify it
Hypothesis: the motion dip is narrow — a checkpoint near step 200 (e.g. 150, 175, 225) exists that
keeps ESR-1/ESR-5 substantially above exp162's step-300 read (60.41/18.47) while clearing the 0.15
motion floor, giving a genuinely usable operating point close to step 200's classification
performance without its guard failure.

Falsified by: no checkpoint between 125 and 275 clears both bars at once — i.e. every checkpoint
near the peak is either at/below exp162's step-300 ESR read, or below the 0.15 motion floor the
same way step 200 was. That would mean the peak is a single isolated point, not a region, and the
void+rank32 combination's only reportable operating points remain the ones already evaluated
(exp164's step 100 or exp162's step 300).

## Setup
`job_type: unlearn`, `frame_replace`, rank 32 / eta 2.0, exp156's void-target dataset — identical
recipe to exp160, only `steps` (600→300) and `save_interval` (100→25) changed. Same
seed/data/hyperparameters mean steps 100/200/300 should reproduce exp164/exp163/exp162's own live
reads exactly, serving as a trajectory sanity check; steps 25/50/75/125/150/175/225/250/275 are new.
Live-monitor-only per exp155's standing practice — a full esr_psr eval on whichever checkpoint the
live trajectory names as the best candidate is the next-tick follow-up, not this run.

## Status
- [ ] Submitted.
- [ ] Live monitor read against exp160's own reads at steps 100/200/300 (sanity check) and the new
      intermediate checkpoints.
- [ ] Candidate checkpoint(s) selected for a full esr_psr eval next tick.
