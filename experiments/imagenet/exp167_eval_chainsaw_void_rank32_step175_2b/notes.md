---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  FALSIFIED: step 175 drops back toward exp162's step-300 read, exactly the falsifier condition.
  Restricted (10-way) row: ESR-1 59.59, ESR-5 18.37, PSR-1 85.40, PSR-5 94.58 — both below
  exp166's step-150 read (72.65/18.37) and ESR-1 is actually the lowest of the whole 100/150/
  175/200/250/300 set, not "one interval below the peak". Erased-class motion is 0.2905, clearing
  the 0.15 floor as predicted. Confirms exp163's step-200 spike is an isolated single-checkpoint
  optimum, not one edge of a plateau — the immediate neighbour on either side (150, 175) both land
  close to step 300's level, well short of step 200's 86.73/43.57.
---
# exp167 — eval: chain-saw void-target dataset x rank 32, CogVideoX-2B, step 175

## Why
Immediate lower neighbour of exp163's step-200 classification peak (thread-best restricted ESR-1
86.73 / ESR-5 43.57, but erased-class motion 0.1379, below GOAL.md's 0.15 floor). See exp166 for
the full rationale; this run evaluates exp165's step-175 checkpoint.

## Hypothesis and what would falsify it
Hypothesis: step 175 retains most of step 200's erasure (ESR-1 in the 70s–80s, ESR-5 in the 30s–40s)
while clearing the 0.15 motion floor — the narrowest test of "usable checkpoint one interval below
the peak".

Falsified by: ESR-1/ESR-5 dropping back toward exp162's step-300 read (60.41/18.47) — the peak is
one isolated checkpoint — OR erased-class motion_score_mean below 0.15.

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp165's `frame_replace_lora_step175`, identical
200-prompt protocol. Submitted alongside exp166 (150), exp168 (225), exp169 (250).

## Status
- [x] Submitted.
- [x] Compared against exp163 (step 200), exp162 (step 300), exp166/exp169 (exp168 still running
      as of this tick).
