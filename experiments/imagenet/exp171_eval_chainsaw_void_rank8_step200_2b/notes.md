---
status: active
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
submitted: 2026-08-27 21:18 helios job 21367439
---
# exp171 — eval: chain-saw void-target dataset x rank 8, CogVideoX-2B, step 200

## Why
See exp170 for the full rationale: exp158 only evaluated exp157's final checkpoint (step 600),
which won on ESR-1/PSR-1/PSR-5 over the random-distractor baseline but left ESR-5 flat. Early
stopping helped ESR-5 substantially at rank 32/64 on the random-distractor dataset but did NOT
rescue it when void-target data was stacked with rank 32 (exp160–exp169). This run tests step 200
on the untested rank-8 + void + early-stopping combination.

## Hypothesis and what would falsify it
Hypothesis: step 200 improves ESR-5 over exp158's step-600 read (15.41), and — with exp170 — traces
whether the gain (if any) grows monotonically toward earlier steps (as it did for rank 64,
exp148→exp149→exp151→exp153) or peaks and reverses (as it did for rank 32 alone, exp150→exp152).

Falsified by: ESR-5 at or below exp158's step-600 read, OR ESR-1/PSR dropping below exp134's
random-distractor baseline.

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp157's `frame_replace_lora_step200`
(pre-existing checkpoint, no new training), identical 200-prompt protocol. Submitted alongside
exp170 (step 300) and exp172 (step 100) — independent evals, no dependency between them.

## Status
- [ ] Submitted.
- [ ] Compared against exp158 (step 600), exp170 (step 300), exp172 (step 100).
