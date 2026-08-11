---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  Sweeps `retention_weight` [0.25, 0.5] on exp104's clothed anchors at eta 2.0, everything else
  identical to exp105 run_002. exp105 measured the endpoints — fire anchors erase but kill motion,
  clothed anchors at weight 1.0 hold motion but never erase — and nothing between them has been
  tried. Asks whether a weight exists that erases while keeping the motion protection. 2 jobs.
---
# exp108 — how much clothed retention?

## Why

[exp105](../exp105_frame_replace_nudity_clothed_retention/notes.md) split cleanly, and both halves
matter:

- **Clothed anchors protect motion.** 2-3x the fire-retention runs over the entire back half of
  training (0.35 vs 0.13 at step 130 against base 0.686), rising where the fire runs keep sliding.
  Ten checkpoints, not the isolated-point noise that has fooled this thread twice. So the motion
  collapse is **not** intrinsic to eta-extrapolated erasure, which is what
  [exp088](../exp088_frame_replace_nudity_clean/notes.md) had left us about to write as a limitation.
- **They also block erasure.** Neither exp105 arm reaches a stable zero; the late phase settles at
  0.29-0.39 against a base of 0.41. Same failure mode as exp085, weaker: even *fully clothed* people
  anchors compete with the erase term, because what they share with the concept is human-body
  features, not wardrobe.

The endpoints of the trade are now measured and nothing in between is:

| anchors | weight | erasure | motion |
|---|---|---|---|
| fire (exp080 r2) | 1.0 | **0.0000** | 0.11 |
| clothed (exp105 r2) | 1.0 | 0.29 late | **0.16-0.35** |
| clothed | **0.5** | ? | ? |
| clothed | **0.25** | ? | ? |

The reason to expect something in the middle is that exp105 is *already* Pareto-better at matched
erasure — at rate 0.02 it reads motion 0.34 / colour 22.7 where exp086 r3 reads 0.16 / 16.8. That
point is a single-step transient and cannot be reported, but it says the frontier moved, and a weight
that lets erasure actually land is what would turn it into a checkpoint.

## One variable

This is exp105 run_002 with `retention_weight` swept and everything else byte-identical: same
dataset, same anchors, same eta, lr, steps, eval budget and seed. The four points sit on one curve.

**eta is fixed at 2.0, not swept.** It was the better of exp105's two arms on every axis that matters
(late-phase rate 0.290 vs 0.394 at the same colour), and sweeping both would spend 4 jobs answering
what 2 jobs answer.

**The contaminated targets are deliberately left in.** exp080's dataset carries 4 of 34 targets that
still trigger the detector — 2 of them at confidence ~0.75 across all 49 frames, i.e. uncorrected
nudity being used as the "concept-removed" target. That is a real defect, found 2026-08-11, and
[exp109](../exp109_split_nudity_gen4_dataset/notes.md) rebuilds around it. Filtering it *here* would
confound the weight sweep with a data change, so this run keeps exactly what exp105 trained on.

## What to watch

- **Erasure and motion together, per checkpoint.** The question is not "does it erase" but whether
  any checkpoint reaches a low rate while motion is still above the ~0.11 the fire anchors leave.
  Plot `nudity_frame_rate` against `motion_score_mean`, not either alone.
- **The late-phase window (steps 160-200), aggregated.** exp105's most interesting behaviour was not
  a minimum but a *plateau*: rate 0.290 +/- 0.076 over 5 checkpoints at near-base colour, where the
  fire arms rebound to 0.44-0.51. Aggregating 5 checkpoints is 2450 frames and far better powered
  than any single n=10 read.
- **Isolated dips are not results.** Three times in this thread a single-checkpoint zero has been
  read as a regime and been contradicted by its neighbours. Require two adjacent checkpoints.
- Per [[feedback-detector-metrics-not-ground-truth]], the winning checkpoint needs human review
  before any number leaves this folder.

## Downstream
If a weight erases while holding motion, that checkpoint replaces exp080 run_002 step 120 as the
reported method row, and [exp102](../exp102_eval_frame_replace_comparable_nudity/notes.md) and
[exp107](../exp107_vbench_utility_frame_replace/notes.md) get repointed at it — both are one-field
changes, so the comparisons survive the swap.

## Status
- [ ] Submitted (2 jobs).
- [ ] Rate-vs-motion frontier plotted against exp105 r2 and exp080 r2.
- [ ] Late-phase (160-200) aggregate compared with exp105 r2's 0.290 +/- 0.076.
- [ ] Human review of the best checkpoint.
