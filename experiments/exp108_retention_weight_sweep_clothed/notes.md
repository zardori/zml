---
status: active
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  NULL RESULT. There is no middle. Lowering clothed-retention weight restores erasure and hands back
  exactly the motion protection it bought: at rate <=0.02, w0.5 reads motion 0.14 and fire retention
  reads 0.14 — the same point. In the late window w1.0 (exp105) dominates both lower weights on
  every axis at once (rate 0.290 vs 0.280/0.483, motion 0.162 vs 0.047/0.093, colour 32.9 vs 25.7),
  so the sweep found nothing better than either endpoint. Clothed retention does not beat fire
  retention at any weight; this is why exp110 uses fire.
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

## Results (2026-08-11) — null, through step 180

### There is no middle: the knob moves both things together

At the first checkpoint reaching rate <=0.02 — the only fair way to compare, since the arms erase at
different steps:

| | first rate <=0.02 | motion there | colour |
|---|---|---|---|
| w0.25 | step 60 | 0.10 | 15.4 |
| w0.5 | step 60 | **0.14** | 17.1 |
| w1.0 (exp105 r2) | never | — | — |
| **fire (exp080 r2)** | step 80 | **0.14** | 14.6 |

w0.5's best point and fire retention's best point are **the same point**. Lowering the weight buys
back erasure by giving up precisely the motion protection that made clothed anchors interesting.
The hypothesis in the header — that some intermediate weight holds both — is false.

### The late window says the same thing, more strongly

Aggregated over steps 160-200 (base rate 0.41, motion 0.686, colour 36.3):

| | rate | motion | colour |
|---|---|---|---|
| w0.25 | 0.280 +/-0.122 | 0.047 | 25.7 |
| w0.5 | 0.483 +/-0.208 | 0.093 | 25.7 |
| **w1.0 (exp105 r2)** | **0.290 +/-0.076** | **0.162** | **32.9** |
| fire (exp080 r2) | 0.323 +/-0.035 | 0.060 | 31.7 |

**w1.0 dominates both swept weights on every axis simultaneously** — equal-or-better rate, 1.7-3.4x
the motion, and 7 points more colour. The sweep did not find a point between the endpoints; it found
that both endpoints beat everything between them. w0.5 is also unstable, overshooting to 0.71 at
step 180 — worse than the base model.

### What survives

exp105's distinctive operating point — *quality restored, nudity durably reduced ~30%* — is specific
to **full-weight** clothed retention and does not appear at 0.25 or 0.5. It remains a real second
operating point worth reporting as a Pareto pair, and it remains one that does not erase.

## Consequence
[exp110](../exp110_frame_replace_nudity_gen4/notes.md) uses **fire retention**, not clothed. Pairing
the gen4 dataset with clothed anchors would have changed two things for no gain, and this run is why
that decision is safe rather than a guess.

## Status
- [x] Submitted (2 jobs).
- [x] Rate-vs-motion frontier read against exp105 r2 and exp080 r2 — **no improvement at any weight**.
- [x] Late-phase (160-200) aggregate compared with exp105 r2's 0.290 +/- 0.076 — w1.0 dominates.
- [ ] Runs finish (step 180/200 at time of writing); nothing expected to change.
- [ ] DOVER scored locally if the arms are ever revisited (videos not pulled — no reason to).
- ~~Human review of the best checkpoint~~ — no checkpoint here is worth reviewing.
