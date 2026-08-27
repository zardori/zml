---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
---
# exp163 — eval: chain-saw void-target dataset x rank 32, CogVideoX-2B, step 200

## Why
exp161 (step 600) and exp162 (step 300) both found that stacking void-target data (exp156) with
rank-32 capacity underperforms EITHER lever alone on ESR-1/ESR-5, even though the two levers each
independently moved a different half of GOAL.md's target (void → ESR-1/PSR, rank32+early-stop →
ESR-5). Step 300 partially recovered from step 600's read (ESR-1 45.71→60.41, ESR-5 10.82→18.47)
but still falls well short of exp150's rank-32-alone step-300 peak (ESR-1 72.76, ESR-5 38.67). This
is an eval-only diagnostic against exp160's already-trained step-200 checkpoint — no new training
required — to see whether the decline-with-training-length trend continues, the way rank 64's
ESR/PSR kept climbing all the way to step 100 (exp149→exp151→exp153) rather than peaking at step
300 the way rank 32 alone did (exp150's peak, exp152's step-200 drop below it).

## Hypothesis and what would falsify it
Hypothesis: step 200 continues the trend from step 600→300 (ESR-1/ESR-5 both higher again),
possibly approaching exp150's rank-32-alone numbers — i.e. void+rank32's interference is a
training-length effect that early stopping can substantially undo, not a hard ceiling.

Falsified by: this checkpoint scoring at or below exp162's step-300 read on ESR-1 or ESR-5 — the
exp152 outcome (step 200 was uniformly worse than step 300 for rank-32-alone), which would mean
step 300 is a local peak for this combination too, mirroring rank 32 alone rather than rank 64.

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp160's `frame_replace_lora_step200` checkpoint,
identical 200-prompt protocol to every other row in this thread. Submitted alongside exp164
(step 100) — independent evals of the same completed training run's saved checkpoints, no
dependency between them.

## Status
- [ ] Submitted.
- [ ] Compared against exp162 (same LoRA, step 300), exp150 (rank 32, random-distractor, step 300)
      and exp164 (same LoRA, step 100).
