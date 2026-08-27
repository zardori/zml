---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
---
# exp170 — eval: chain-saw void-target dataset x rank 8, CogVideoX-2B, step 300

## Why
exp158 reported only exp157's final checkpoint (step 600, rank 8, void-target dataset): restricted
ESR-1 70.92, ESR-5 15.41, PSR-1 85.95, PSR-5 95.44. That is a clean win over the random-distractor
baseline (exp134: 49.90/15.61/82.71/93.19) on ESR-1 and both PSR cells, but ESR-5 is flat (-0.20,
inside noise) — the residual top-5 problem every other single lever has struggled with.

Early stopping is the one lever that has moved ESR-5 substantially elsewhere in this thread, but
always on the random-distractor dataset: rank 32 (exp143 step-600 20.92 → exp150 step-300 38.67)
and rank 64 (exp148 step-600 16.43 → exp153 step-100 44.49). It has never been tried on the
void-target dataset at rank 8 — exp157/exp158 only ever evaluated the final checkpoint. Separately,
when void-target data was stacked with rank 32 (exp160–exp169), early stopping did NOT rescue
ESR-5 the way it did on the random-distractor dataset — the two levers interfered rather than
adding. Whether that interference is specific to rank 32 or extends to rank 8 (where void alone
already won cleanly) is the open question this run and exp171/exp172 answer.

## Hypothesis and what would falsify it
Hypothesis: step 300 improves ESR-5 over exp158's step-600 read (15.41) without giving back the
ESR-1/PSR gains void bought — mirroring the random-distractor early-stop pattern, not the
void+rank-32 interference pattern.

Falsified by: ESR-5 at or below exp158's step-600 read (15.41), OR ESR-1/PSR-1/PSR-5 dropping
below exp134's random-distractor baseline (49.90/82.71/93.19) — either would mean early stopping
does not help this combination the way it helped rank 32/64 alone, and that void+capacity
interference (exp160–exp169) generalizes to void+early-stopping too.

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp157's `frame_replace_lora_step300`
(pre-existing checkpoint, no new training), identical 200-prompt protocol to every other row in
this thread. Submitted alongside exp171 (step 200) and exp172 (step 100) — independent evals, no
dependency between them.

## Status
- [ ] Submitted.
- [ ] Compared against exp158 (step 600), exp171 (step 200), exp172 (step 100).
