---
status: active
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run. Late-training reference (step 300) for exp173's rank-16 void-target LoRA — anchors
  where the curve settles after the step-200 peak and confirms the peak is mid-training, not a climb.
submitted: 2026-08-28 11:01 helios job 21388532
---
# exp177 — eval: chain-saw void-target dataset x rank 16, CogVideoX-2B, step 300

## Why
Step 300 is the final checkpoint of exp173's fine-grained run. Both other ranks on the void-target
dataset showed a mid-training ESR-5 peak with step 300 sitting below it (rank 8: exp170 30.20 vs
exp171's step-200 32.04 — barely; rank 32: exp162 18.47 vs exp163's step-200 43.57 — far). This run
anchors rank 16's post-peak level so the exp174/exp175/exp176 bracket has a converged reference,
and confirms rank 16 also has a mid-training peak (step 200 > step 300) rather than a curve still
rising at step 300.

## Hypothesis and what would falsify it
Hypothesis: step 300 ESR-5 is below exp175's step-200 read (rank 16, like both other ranks on this
dataset, peaks mid-training), and above exp158's rank-8 step-600 floor (15.41).

Falsified by: step 300 ESR-5 at or above exp175's step-200 read — no mid-training peak for rank 16,
the curve is still climbing at the end of the run and a longer budget is warranted.

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp173's pre-existing `frame_replace_lora_step300`,
identical 200-prompt protocol. Submitted alongside exp174 (150), exp175 (200), exp176 (250) —
independent, no dependency between them.

## Status
- [ ] Submitted.
- [ ] Compared against exp174/exp175/exp176 and the rank-8 / rank-32 void curves.
