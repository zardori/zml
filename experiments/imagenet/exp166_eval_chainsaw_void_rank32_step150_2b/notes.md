---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  PARTIAL FALSIFICATION: ESR-1 clears the bar, ESR-5 does not. Restricted (10-way) row: ESR-1
  72.65, ESR-5 18.37, PSR-1 85.20, PSR-5 96.01. Against exp162's step-300 read (60.41/18.47):
  ESR-1 is meaningfully above (+12.24) but ESR-5 is flat (-0.10, inside noise) — the peak has
  NOT risen at step 150 on the metric that matters most. Erased-class motion is 0.2367, clearing
  the 0.15 floor comfortably (as predicted). So step 150 is not "step 200 minus a bit" — it sits
  close to step 300's level on ESR-5, confirming exp163's step-200 read is a narrow, isolated
  spike rather than the edge of a wider plateau. See exp167/exp169 for the same conclusion from
  the other side.
---
# exp166 — eval: chain-saw void-target dataset x rank 32, CogVideoX-2B, step 150

## Why
exp163 found exp160's step-200 checkpoint (void-target data x rank 32) is this thread's best
classification row (restricted ESR-1 86.73, ESR-5 43.57) but the first fully-evaluated checkpoint
to breach GOAL.md's 0.15 erased-class motion floor (0.1379). The 100-step-resolution sweep
(exp161–exp164) cannot tell whether that motion dip is a narrow transient with a usable checkpoint
beside it, or the edge of a wider collapse. exp165 re-trained the identical recipe at
save_interval 25; this run evaluates its step-150 checkpoint — the lower bracket of the peak.

## Hypothesis and what would falsify it
Hypothesis: step 150 keeps ESR-1/ESR-5 meaningfully above exp162's step-300 read (60.41/18.47)
while clearing the 0.15 motion floor — a usable operating point on the rising edge of the peak.

Falsified by: ESR-1 or ESR-5 at or below exp162's step-300 read (the peak has not yet risen at
step 150, i.e. it is narrower than ±50 steps), OR erased-class motion_score_mean below 0.15 (the
motion breach extends to the rising edge, not just step 200).

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp165's `frame_replace_lora_step150`, identical
200-prompt protocol to every other row in this thread. Submitted alongside exp167 (175), exp168
(225), exp169 (250) — independent evals of exp165's saved checkpoints.

## Status
- [x] Submitted.
- [x] Compared against exp163 (step 200), exp162 (step 300), exp167/exp169 (exp168 still running
      as of this tick).
