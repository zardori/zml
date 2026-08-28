---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run. Lower bracket (step 150) of exp173's rank-16 void-target LoRA, testing whether rank
  16's ESR-5 peak sits below step 200 or the peak is centred there like rank 8 and rank 32.
---
# exp174 — eval: chain-saw void-target dataset x rank 16, CogVideoX-2B, step 150

## Why
exp173 trained exp156's void-target chain-saw dataset at rank 16 — the untested midpoint between
rank 8 (exp157/exp158/exp170–172) and rank 32 (exp160/exp165–169), both of which peak their
restricted ESR-5 at step 200 on this dataset:

- rank 8, step 200 (exp171): ESR-1 68.88 / ESR-5 32.04 / PSR-1 82.06 / PSR-5 94.69, motion 0.498 — legal.
- rank 32, step 200 (exp163): ESR-1 86.73 / ESR-5 43.57 / PSR-1 82.95 / PSR-5 93.45, motion 0.1379 — **breaches** the 0.15 motion guard.

So capacity appears to raise both the peak's ESR-5 amplitude and its motion risk. Rank 16 is the
test of whether a legal peak bigger than rank 8's exists in between. exp173's live monitor
converged by step 75 but its 9-prompt motion signal was too noisy to pick a checkpoint, so this is
one of a four-eval bracket (150 / 200 / 250 / 300).

## Hypothesis and what would falsify it
Hypothesis: step 150 sits below rank 16's step-200 ESR-5 reading (the peak is centred at 200, as it
is for both other ranks), so this run's restricted ESR-5 comes in under exp175's.

Falsified by: ESR-5 at step 150 at or above exp175's step-200 read — the rank-16 peak has shifted
earlier — or ESR-5 already collapsed to exp158's step-600 level (15.41), meaning rank 16 shows no
mid-training peak at all.

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp173's pre-existing `frame_replace_lora_step150`,
identical 200-prompt protocol. Submitted alongside exp175 (200), exp176 (250), exp177 (300) —
independent, no dependency between them.

## Status
- [ ] Submitted.
- [ ] Compared against exp175/exp176/exp177 and the rank-8 (exp170–172) / rank-32 (exp163–169) curves.
