---
status: active
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Reported ESR/PSR row for the chain-saw LoRA on the same 200 prompts and seeds as
  exp064/exp065.
---
# exp071 — reported ESR/PSR for the chain-saw LoRA

## Goal
Produce the `frame_replace (ours)` row for chain saw, on the same 200 prompts and seeds as exp064
(Original) and exp065 (NegPrompt). This is the number that goes into the comparison table; exp069's
live eval was a progress signal on a 9-prompt collateral sample, not the metric.

## Setup
`mode: imagenet`, `erased_class: "chain saw"`, `lora_checkpoint_dir` pointing at the exp069
checkpoint being reported. Generation covers all ten classes (the erased one for ESR, the other nine
for PSR); everything else matches exp064 exactly, so the rows are comparable.

Pick the checkpoint deliberately and record which one here — reporting the best of six checkpoints
selected on the eval set would be selection on the test set. Default to the final step unless there
is a stated reason.

`./submit_job.py athena experiments/exp071_eval_frame_replace_chainsaw/config.yaml`

## What to watch
- ESR-1 / ESR-5 vs. exp064 and exp065 for chain saw.
- **ESR-5 in particular.** T2VUnlearning's central claim is that baselines raise ESR-1 while leaving
  ESR-5 low, because they distort the object rather than remove it; a high ESR-5 is the evidence of
  actual removal. Ours landing high on ESR-1 but low on ESR-5 would mean the same weakness.
- PSR-1 / PSR-5 vs. exp064: how much of the other nine classes survived. The paper trades PSR-1 down
  from 78.38 (Original) to 54.03 for its erasure, so some loss is expected and normal.
- `quality` block per class against exp064's, to catch a general quality collapse hiding behind a
  good ESR.

## Status
- [ ] exp069 complete; checkpoint chosen and recorded here.
- [ ] Submitted.
- [ ] Row added via `tools/build_imagenet_table.py`.
