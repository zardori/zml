---
status: ready
concept: nudity
method: eval
thread: nudity
takeaway: >
  T2VUnlearning's two utility dimensions re-measured on exp110's checkpoint, pairing against exp106
  (base, unchanged). Note the honest tension: exp110 holds far more motion, so the Subject
  Consistency demonstration (+2.17 while motion -36%) should get SMALLER — a weaker metric critique
  but a better method. Both go in the paper. 2 jobs.
---
# exp113 — VBench utility on the gen4 checkpoint

## Why
[exp107](../exp107_vbench_utility_frame_replace/notes.md) measured these on exp080 r2 s120 and
produced two results the paper leans on:

1. **Subject Consistency 96.41 vs base 94.24 (+2.17)** while motion fell 36% on the same clips —
   their preservation metric rewarding our worst failure mode.
2. **Colorfulness preserved off-concept** (-8.5% / +8.2%) while dropping 40% on-concept — evidence
   the colour loss is the erasure operating, not general degradation.

[exp110](../exp110_frame_replace_nudity_gen4/notes.md) changes the checkpoint under both. It holds
2.3x the motion on concept prompts and nearly all its colour.

## What to expect, written down before the run
**The Subject Consistency gain should shrink.** That metric rewards stillness, so a less frozen
checkpoint gains less on it. If exp110 scores nearer base — or below it — the demonstration is
weaker in magnitude while the method is better. Both facts belong in the write-up, and the
relationship between them (gain scales with how static the clip is) is itself the argument, already
visible in exp107's own data: +4.98 on near-frozen I2P clips vs +2.17 on the livelier VBench set.

Recording this now so a smaller number reads as a confirmed prediction rather than a disappointment.

## Setup
Same two prompt sets, same seeds, so it pairs directly against exp106 (base) with no re-run needed.
Scoring is post-hoc and local: `tools/score_subject_consistency.py`, `tools/score_dover.py`.
Object Class still has no instrument (GRiT/detectron2 or an OWL-ViT substitute) — unchanged here.

## Status
- [ ] Submitted (2 jobs).
- [ ] Subject Consistency scored against exp106's 94.24 base and exp107's 96.41.
- [ ] Colorfulness on/off-concept compared against exp107's -8.5% / +8.2%.
- [ ] Object Class instrument still missing.
