---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  THE REPORTED ROW, and it cuts both ways. Erasure is the best in the table under either convention:
  1000-way ESR-1 100.0 / ESR-5 89.8 against NegPrompt's 70.9 / 44.4 and base's 48.7 / 20.9;
  restricted to the ten protocol classes ESR-1 49.0 / ESR-5 10.0 against NegPrompt's 17.2 /
  0.00. Preservation costs almost nothing more than NegPrompt (PSR-1 52.5 vs 53.1, restricted 83.8 vs
  83.4). BUT the pre-registered quality check FAILS: the other nine classes lose a mean 45% of
  their motion (cassette player -78%, gas pump -78%, French horn -75%) against chain saw's -80%, so
  the freeze is NOT concept-conditional as exp069's live monitor suggested — it is global, and the
  row must carry that caveat. Colorfulness is up on all ten classes. Same lesson as exp112 in the
  nudity thread: a small live-eval set flattered the result.
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

## Results (2026-08-17) — best erasure in the table, bought with a global freeze

Completed on athena in 13.6 h (200 prompts, 10 classes, `frame_replace_lora_step600`).

### The row, under both conventions

| chain saw | ESR-1 | ESR-5 | PSR-1 | PSR-5 |
|---|---|---|---|---|
| base (exp064), 1000-way | 48.67 | 20.92 | 55.40 | 75.77 |
| NegPrompt (exp065), 1000-way | 70.92 | 44.39 | 53.07 | 71.58 |
| **frame_replace, 1000-way** | **100.00** | **89.80** | 52.47 | 72.63 |
| base (exp064), restricted | 5.41 | 0.71 | 89.59 | 96.26 |
| NegPrompt (exp065), restricted | 17.24 | 0.00 | 83.36 | 93.55 |
| **frame_replace, restricted** | **48.98** | **10.00** | 83.75 | 91.62 |

Per-class chain saw: 1000-way top-1 0.5133 -> **0.0000**, top-5 0.7908 -> **0.1020**; restricted
top-1 0.9459 -> **0.5102**, top-5 0.9929 -> **0.9000**.

### ESR-5 is the test, and the answer depends on the convention

T2VUnlearning's central claim is that baselines raise ESR-1 while leaving ESR-5 low, because they
distort the object rather than remove it. Under the **1000-way** convention we pass that test
outright — ESR-5 89.80, i.e. the classifier does not place chain saw in its top five for 90% of the
clips, against NegPrompt's 44.39. That is removal, not distortion.

Under the **restricted** convention the same checkpoint shows the very signature the paper criticises:
ESR-1 up 9x over base (5.41 -> 48.98) while ESR-5 reaches only 10.00. Given ten choices, the residual
still ranks chain saw in the top five for 90% of clips.

Both are true of one checkpoint. This is a sharper version of exp065's finding — there the split was
70.9 vs 17.2 for a baseline; here it is 100.0 vs 49.0 for *our own headline*. **Neither number may be
reported without naming its convention**, and `docs/imagenet_objects.md`'s two-convention rule is now
load-bearing for our own row, not just for NegPrompt's.

### Preservation is cheap; motion is not

PSR barely moves against NegPrompt (52.47 vs 53.07 1000-way, 83.75 vs 83.36 restricted) and costs
about 3 points against base 1000-way and 6 restricted. So the *semantics* of the other nine classes
survive: the classifier still recognises them.

Their **motion** does not. This is the pre-registered check from "What exp069 predicts", and it fails:

| class | base motion | ours | delta |
|---|---|---|---|
| chain saw *(erased)* | 0.563 | 0.111 | **-80%** |
| cassette player | 0.688 | 0.149 | -78% |
| gas pump | 0.377 | 0.082 | -78% |
| French horn | 0.487 | 0.124 | -75% |
| church | 0.481 | 0.210 | -56% |
| parachute | 0.341 | 0.205 | -40% |
| garbage truck | 0.663 | 0.407 | -39% |
| tench | 0.997 | 0.779 | -22% |
| English springer | 1.419 | 1.147 | -19% |
| golf ball | 0.496 | 0.491 | -1% |

**Mean over the nine preserved classes: -45%**, against the erased class's -80%. The header's rule
was explicit: *"If every class's motion collapses, the ESR was bought with degeneration and the row
must be reported with that caveat."* It applies. Colorfulness rises on all ten classes (+7% to +61%),
so the over-saturation is global too; clip score is essentially unharmed everywhere (0.29-0.34).

NegPrompt, for contrast, runs chain-saw motion at **1.114** — *above* base 0.563. It does not freeze
anything. On quality it beats us clearly, and that has to be stated alongside the ESR win.

### This overturns exp069's "concept-conditional" reading

exp069's live monitor showed the unrelated set losing only ~30% of its motion and concluded the
freeze was concept-conditional, unlike nudity's global collapse (exp107, exp111). On the real
200-prompt protocol that does not hold: seven of the nine preserved classes lose 19-78%, and three of
them lose nearly as much as the erased class. The live monitor's unrelated set was too small and too
unlike the protocol classes to see it.

That is the same failure exp112 found in the nudity thread the same week — a 10-prompt live eval
producing a result the full set does not support. **Live-eval sets are progress signals, not
findings**, in both threads.

## Status
- [x] exp069 complete (2026-08-16, 600/600 steps); checkpoint chosen and recorded above.
- [x] Submitted and complete (athena, 13.6 h, `outputs_20260816_222224`).
- [x] Row measured under both conventions; quality block checked against exp064.
- [ ] Row added via `tools/build_imagenet_table.py`.
- [ ] `docs/imagenet_objects.md` updated: the two-convention rule now applies to our own row, and
      exp069's concept-conditional-freeze claim needs correcting there.
- [ ] Human review of chain-saw and cassette-player clips — confirm the -78% on a *preserved* class
      looks like the same "static poster" failure exp069 described.
