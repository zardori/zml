---
status: done
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  CONVERGED, LIVE MONITOR CANNOT RESOLVE THE MOTION QUESTION — HANDED TO THE FULL-EVAL BRACKET
  (exp166–exp169). Classification trajectory reproduces: 9-prompt concept top-1/top-5 are both at
  0.00 from step 175 onward (one 1-video blip at step 150, top-1 0.05) and near floor from step 75,
  matching exp164/exp163/exp162's own top-1/top-5 0.00 reads at steps 100/200/300 — so
  save_interval does not perturb the training trajectory, as exp155 found, and the step-200
  classification peak exp163 measured is a property of this recipe, not of exp160's specific run.
  The live MOTION sample does NOT reproduce exp160's: concept motion here is 0.700 (step 100,
  matches exp160's 0.712) but 0.214 (step 200, vs exp160's live 0.085 and exp163's full-protocol
  0.138) and 0.417 (step 300, vs exp160's live 0.202) — the 9-prompt motion signal is too noisy
  across runs to reproduce even its own prior read, let alone pick a checkpoint. Across the fine
  grid concept motion dips to 0.142–0.217 over steps 125–200 then recovers to 0.29–0.42 by
  steps 225–300, consistent with (but not confirming) exp163's finding that the motion breach is a
  narrow dip coincident with the ESR peak rather than a sustained collapse. Per the pre-registered
  plan, the full 200-prompt esr_psr evals on the bracketing checkpoints (exp166 step 150, exp167
  step 175, exp168 step 225, exp169 step 250) are what actually answer whether a usable operating
  point exists near step 200 — this run only produced the checkpoints and ruled out
  non-convergence.
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
- [x] Submitted (helios, job 21324328, 14341s).
- [x] Live monitor read. Classification sanity check PASSES (top-1/top-5 0.00 at 100/200/300 as in
      exp164/exp163/exp162); live motion sample does NOT reproduce exp160's (0.214 vs 0.085 at
      step 200) — 9-prompt motion is too noisy to select on.
- [x] Candidates selected: steps 150 / 175 / 225 / 250, evaluated on the full protocol as
      exp166 / exp167 / exp168 / exp169.
