---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  PARTIAL FALSIFICATION: ESR-1 is above exp162's step-300 read but ESR-5 is BELOW it, so the
  hypothesized monotonic recovery does not hold. Restricted (10-way) row: ESR-1 66.94, ESR-5
  15.00, PSR-1 84.80, PSR-5 94.83. Against exp162 (step 300: 60.41/18.47): ESR-1 up (+6.53) but
  ESR-5 down (-3.47) — step 250 is not "between step 200 and step 300" on ESR-5, it is the lowest
  ESR-5 reading of the whole 100/150/175/200/250/300 sweep. Erased-class motion is 0.3645,
  clearing the 0.15 floor with a healthy margin, close to step 300's 0.408 as expected. With
  exp166/exp167, this closes the practical question the fine-grained sweep was built to answer:
  no checkpoint within ±50 steps of exp163's step-200 spike reproduces more than a fraction of its
  erasure — 150/175/250 all cluster in the ESR-1 60-73 / ESR-5 15-18 band, matching step 300's
  level, while only step 200 itself reaches 86.73/43.57 and only step 200 breaches the motion
  floor. The spike is a single isolated checkpoint, not a rescuable plateau; exp150's rank-32
  step-300 early-stop optimum on the random-distractor dataset (72.76/38.67, motion 0.296)
  remains a cleaner and better full-protocol row than anything on this void+rank-32 LoRA.
---
# exp169 — eval: chain-saw void-target dataset x rank 32, CogVideoX-2B, step 250

## Why
Outer upper bracket of exp163's step-200 classification peak, halfway to exp162's step-300 read
(60.41/18.47, motion 0.408). With exp168 (225) it fixes the decay rate of the ESR gain on the
trailing edge and the step at which the 0.15 motion floor is cleared. See exp166 for the full
rationale; this run evaluates exp165's step-250 checkpoint.

## Hypothesis and what would falsify it
Hypothesis: step 250 sits between step 200's erasure and step 300's — ESR-1/ESR-5 above exp162's
step-300 read, motion above 0.15 — confirming a monotonic recovery of both metrics from the peak
toward step 300.

Falsified by: ESR-1/ESR-5 at or below exp162's step-300 read at step 250 (the ESR gain is gone
50 steps past the peak) — which, with exp168, would show the peak has no usable region on the
trailing edge either.

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp165's `frame_replace_lora_step250`, identical
200-prompt protocol. Submitted alongside exp166 (150), exp167 (175), exp168 (225).

## Status
- [x] Submitted.
- [x] Compared against exp163 (step 200), exp162 (step 300), exp166/exp167 (exp168 still running
      as of this tick).
