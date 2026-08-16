---
status: done
concept: nudity
method: eval
thread: nudity
takeaway: >
  Subject Consistency 96.75 (old checkpoint 96.41, base 94.24) — the metric critique is INTACT and
  marginally stronger. The prediction that it would shrink was not tested: motion on this set barely
  moved (1.06 vs 1.03), so the input to the mechanism did not change. Colour is now ABOVE base on
  both sets (53.1 vs 45.8, 49.6 vs 40.5) and object_class motion recovers from -68% to -49%. So on
  general content the gen4 checkpoint is clearly gentler.
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


## Results (2026-08-14)

Checkpoint exp110 step 140, paired against exp106 (base) and exp107 (old checkpoint, exp080 r2 s120):

| set | metric | base | old | **new** |
|---|---|---|---|---|
| `object_class` (79) | clip | 0.28 | 0.28 | 0.28 |
| | colorfulness | 45.79 | 41.92 | **53.12** |
| | motion | 0.92 | 0.29 (-68%) | **0.47 (-49%)** |
| | nudity rate | 0.053 | 0.010 | 0.030 |
| `subject_consistency` (72) | clip | 0.31 | 0.30 | 0.31 |
| | colorfulness | 40.46 | 43.76 | **49.57** |
| | motion | 1.60 | 1.03 (-36%) | 1.06 (-34%) |
| | **Subject Consistency** | **94.24** | **96.41** | **96.75** |

### The metric critique survives, marginally stronger

**96.75 against a base of 94.24 — a +2.51 gain while motion is down 34% on the same clips**, versus
T2VUnlearning's own -0.83 on this metric. The demonstration that their preservation instrument
rewards temporal collapse is unchanged by the better checkpoint.

**The prediction written here before the run was not tested.** It said the gain should *shrink*
because a less-frozen checkpoint scores lower. But motion on this particular set barely moved
(1.06 vs 1.03), so the mechanism's input never changed and neither did the output. That is a
non-result, not a confirmation — the prediction remains untested and should not be reported as
supported. exp107's within-checkpoint evidence (+4.98 on near-frozen I2P clips vs +2.17 on livelier
VBench ones) is still the only support for the stillness-scaling claim.

### Colour above base is worth a second look
53.12 vs 45.79 and 49.57 vs 40.46 — the gen4 checkpoint is **more colourful than the base model** on
content unrelated to nudity. Combined with exp112's colour gains on every concept set, the gen4
training appears to push saturation globally rather than merely stop destroying it. Harmless for the
preservation story, but it is a distribution shift on unrelated content and should be named as one
rather than reported as "preserved".

## Status
- [x] Submitted and complete (2 jobs).
- [x] Subject Consistency scored: **96.75** vs base 94.24 and old checkpoint 96.41.
- [x] Colorfulness on/off-concept compared — now above base on both unrelated sets.
- [ ] DOVER scored locally on these outputs.
- [ ] Object Class instrument still missing (GRiT/detectron2 or OWL-ViT).
