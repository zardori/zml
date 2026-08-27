---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
---
# exp172 — eval: chain-saw void-target dataset x rank 8, CogVideoX-2B, step 100

## Why
See exp170 for the full rationale. This is the earliest checkpoint exp157 saved. exp157's live
9-prompt monitor showed concept top-1 already at 0.00 by step 100 (faster than exp133's
identical-recipe/random-distractor run, which took until step 200), so step 100 is not obviously
undertrained despite being the earliest available checkpoint — the same reasoning that made rank
64's step-100 checkpoint (exp153) worth evaluating and it turned out to be that thread's best
full-protocol row.

## Hypothesis and what would falsify it
Hypothesis: step 100 continues any ESR-5 gain seen at exp170 (step 300) / exp171 (step 200) rather
than reversing it — mirroring rank 64's monotonic climb all the way to step 100 (exp153), not rank
32's single non-monotonic peak (exp150 vs exp152/exp154).

Falsified by: ESR-5 dropping back toward or below exp158's step-600 read (15.41) after rising at
step 300/200, indicating a peak-and-reverse shape instead of a monotonic one — or by ESR-5 never
having risen at exp170/exp171 in the first place, in which case this checkpoint has nothing left
to falsify beyond confirming the flat/null result.

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp157's `frame_replace_lora_step100`
(pre-existing checkpoint, no new training), identical 200-prompt protocol. Submitted alongside
exp170 (step 300) and exp171 (step 200) — independent evals, no dependency between them.

## Status
- [ ] Submitted.
- [ ] Compared against exp158 (step 600), exp170 (step 300), exp171 (step 200).
