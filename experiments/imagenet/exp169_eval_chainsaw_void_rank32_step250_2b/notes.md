---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
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
- [ ] Submitted.
- [ ] Compared against exp163 (step 200), exp162 (step 300), exp168 (step 225).
