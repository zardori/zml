---
status: active
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run. Primary run of the exp174-exp177 bracket: reads rank 16's step-200 peak ESR-5 and
  whether erased-class motion there clears the 0.15 guard (rank 8 did, rank 32 did not).
submitted: 2026-08-28 11:01 helios job 21388529
---
# exp175 — eval: chain-saw void-target dataset x rank 16, CogVideoX-2B, step 200

## Why
Both ranks already measured on exp156's void-target chain-saw dataset peak their restricted ESR-5
at step 200:

- rank 8, step 200 (exp171): ESR-1 68.88 / ESR-5 32.04 / PSR-1 82.06 / PSR-5 94.69, motion 0.498 — legal.
- rank 32, step 200 (exp163): ESR-1 86.73 / ESR-5 43.57 / PSR-1 82.95 / PSR-5 93.45, motion 0.1379 — **breaches** the 0.15 motion guard.

Rank 16 (exp173) is the direct test of whether the capacity/motion-risk trade between those two is
continuous — a legal ESR-5 peak bigger than rank 8's — or a step function where any capacity
increase over rank 8 either does nothing or breaches the floor. This is the checkpoint that answers
it: rank 16's peak amplitude, and its erased-class `motion_score_mean` at that peak.

## Hypothesis and what would falsify it
Hypothesis: rank 16 step 200 gives restricted ESR-5 between rank 8's 32.04 and rank 32's 43.57,
with erased-class motion clear of the 0.15 floor (unlike rank 32) — a legal row that improves on
exp150's rank-32/step-300 random-distractor best (ESR-5 38.67, motion 0.296) or at least on rank 8
void.

Falsified by: ESR-5 at or below rank 8's 32.04 (capacity buys nothing on this dataset), OR
erased-class motion below 0.15 (the trade is a step function — rank 16 already inherits rank 32's
motion breach).

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp173's pre-existing `frame_replace_lora_step200`,
identical 200-prompt protocol. Submitted alongside exp174 (150), exp176 (250), exp177 (300) —
independent, no dependency between them.

## Status
- [ ] Submitted.
- [ ] Compared against exp174/exp176/exp177, exp171 (rank 8 step 200), exp163 (rank 32 step 200),
  and GOAL.md's target/guards.
