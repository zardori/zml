---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
---
# exp168 — eval: chain-saw void-target dataset x rank 32, CogVideoX-2B, step 225

## Why
Immediate upper neighbour of exp163's step-200 classification peak (thread-best restricted ESR-1
86.73 / ESR-5 43.57, erased-class motion 0.1379 < 0.15 floor). The trailing edge is where motion
recovers with more training (exp162's step-300 motion was 0.408), so it is the most likely place
for a checkpoint that keeps the erasure and clears the floor. See exp166 for the full rationale;
this run evaluates exp165's step-225 checkpoint.

## Hypothesis and what would falsify it
Hypothesis: step 225 retains most of step 200's erasure (ESR-1 70s–80s, ESR-5 30s–40s) while
clearing the 0.15 motion floor — a usable operating point on the peak's recovering edge.

Falsified by: ESR-1/ESR-5 already collapsed toward exp162's step-300 read (60.41/18.47) — the peak
is narrower than ±25 steps — OR erased-class motion_score_mean still below 0.15 at step 225.

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp165's `frame_replace_lora_step225`, identical
200-prompt protocol. Submitted alongside exp166 (150), exp167 (175), exp169 (250).

## Status
- [ ] Submitted.
- [ ] Compared against exp163 (step 200), exp162 (step 300), and the other brackets.
