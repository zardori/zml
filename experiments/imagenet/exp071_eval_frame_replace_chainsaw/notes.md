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

**Checkpoint reported: `outputs_20260816_003333/frame_replace_lora_step600`** — exp069's final step,
the default. There is no reason to deviate: on the live monitor exp069 is at concept top-1 0.00 from
step 200 through step 600, so every candidate is tied on erasure and choosing among them could only
be a choice about collateral, made on the set the row is reported on.

`slurm_time` raised 10 h → 14 h: exp065 timed out at 163/200 in 10 h on these exact 200 prompts.

`./submit_job.py athena experiments/imagenet/exp071_eval_frame_replace_chainsaw/config.yaml`

## The bar, from exp064 (base) and exp065 (NegPrompt)

| chain saw | ESR-1 | ESR-5 | PSR-1 | PSR-5 | motion | DOVER tech |
|---|---|---|---|---|---|---|
| base, 1000-way | 48.67 | 20.92 | 55.40 | 75.77 | 0.563 | 0.100 |
| base, restricted | 5.41 | 0.71 | 89.59 | 96.26 | | |
| NegPrompt, 1000-way | 70.92 | 44.39 | 53.07 | 71.58 | 1.114 | 0.094 |
| NegPrompt, restricted | **17.24** | **0.00** | 83.36 | 93.55 | | |

Read the restricted row before claiming a win over NegPrompt. Its 1000-way ESR of 70.9 is mostly
sibling-class confusion; ranked within the ten protocol classes the same defence erases essentially
nothing (ESR-5 0.00). Both rows were re-scored locally on 2026-08-16 so all three exist under both
conventions and with DOVER — the numbers above are what is on disk, not the pre-rescore values.

Note NegPrompt's motion (1.11 against base 0.563): whatever it does, it does not freeze the clips.
That is the comparison exp069's checkpoint has to survive.

## What exp069 predicts, and what would contradict it

exp069's 9-prompt monitor says top-1 0.00 / top-5 0.27 with clip score at base level (0.32 vs 0.322),
so ESR should come in high under both conventions. The thing to check is not ESR but the `quality`
block: exp069 froze its concept clips (motion 0.010 against a base of 0.564) while leaving the nine
other classes at −30%. If PSR holds and the other nine classes' motion is near base, the freeze is
concept-conditional and the row is honest. If every class's motion collapses, the ESR was bought with
degeneration and the row must be reported with that caveat.

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
- [x] exp069 complete (2026-08-16, 600/600 steps); checkpoint chosen and recorded above.
- [ ] Submitted.
- [ ] Row added via `tools/build_imagenet_table.py`.
