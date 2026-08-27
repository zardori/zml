---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
---
# exp164 — eval: chain-saw void-target dataset x rank 32, CogVideoX-2B, step 100

## Why
Companion to exp163 (step 200) — see its notes.md for the full rationale. exp161 (step 600) and
exp162 (step 300) show void+rank32 underperforming both single levers on ESR-1/ESR-5, with a
partial recovery from step 600 to step 300. This is the earliest saved checkpoint from exp160's
run, testing the far end of the training-length trend in one tick alongside exp163 rather than
waiting for exp163's result first — the two evals are independent (same completed training run,
different saved checkpoints, no dependency between them).

## Hypothesis and what would falsify it
Hypothesis: step 100 continues or plateaus the step-600→300 trend, mirroring rank 64's monotonic
climb all the way to step 100 (exp149→exp151→exp153) rather than rank 32 alone's non-monotonic
curve (peak at step 300, exp150; decline at step 200, exp152; partial recovery at step 100,
exp154). exp160's live monitor already showed top-1 reaching 0.00 by step 100 (faster than rank
32's random-distractor run, which took until step 200) and step 100 has this run's healthiest live
motion read (0.712) — consistent with void-target data accelerating convergence the way exp157/158
found for rank 8.

Falsified by: this checkpoint scoring at or below exp163's step-200 read on ESR-1 or ESR-5 (the
trend reverses before reaching the earliest checkpoint, mirroring rank 32 alone's own non-monotonic
shape rather than rank 64's monotonic one).

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp160's `frame_replace_lora_step100` checkpoint,
identical 200-prompt protocol to every other row in this thread.

## Status
- [ ] Submitted.
- [ ] Compared against exp163 (same LoRA, step 200), exp154 (rank 32, random-distractor, step 100)
      and exp153 (rank 64, random-distractor, step 100, this thread's overall best row).
