---
status: ready
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  Attacks the one defect exp069 left: chain saw erases completely (top-1 0.506 -> 0.00) but the
  concept clips freeze (motion 0.010 vs base 0.564). Sweeps erase_esd_eta [1.0, 1.5, 2.0] at 300
  steps on the enlarged 33-row gen2 dataset, with 2.0 reproducing exp069's setting so eta and dataset
  size stay separable. Gate: an arm that holds top-1 at 0.00 with motion above ~0.2. Not submitted yet.
---
# exp123 — does weaker erase pressure buy back the motion?

## The question
exp069 is the pilot's positive result and its problem in one run. Chain saw erases — top-1 0.00 from
step 200 on the full-object eval prompts, with the workshop scene intact around where the saw was.
But every concept clip is a still image: motion 0.010 against a base of 0.564, already at step 100,
with colorfulness up 40% and clip score unchanged. A frozen clip is not a usable model, and reporting
an ESR that might have been bought by degeneration is exactly the criticism T2VUnlearning levels at
its baselines.

`erase_esd_eta` sets how far the training target extrapolates **past** the concept-removed donor. At
2.0 (exp069, inherited from exp062) the target is not "the donor" but "twice as far from the concept
as the donor" — the strongest setting anyone here has run. That is the natural first suspect for a
model that responds by refusing to render anything at all, and exp086 measured a real trade-off along
this axis for nudity (eta 1.5 erased with less degeneracy than 2.0, though human review still
preferred 2.0's clips).

## Two hypotheses this separates

The freeze is concept-conditional — exp069's unrelated set only lost 30% of its motion where the
concept set lost 98% — which leaves two readings:

1. **Pressure.** The erase term is strong enough that the cheapest way to satisfy it is to stop
   generating. Lower eta should then relax the freeze *before* it costs erasure.
2. **Substitution.** The LoRA has learned "chain-saw prompt → still life", i.e. the frozen quality is
   part of what it learned to draw instead of the object. Lower eta then buys motion only by giving
   back the object, and the two move together at every setting.

Reading 1 shows up as an arm with top-1 0.00 and motion well above exp069's 0.010; reading 2 shows up
as motion and top-1 rising together across arms, with no separation. Either way the answer is worth
the three jobs, and reading 2 would redirect the fix to the retention branch (`retention_weight`, or
motion-carrying anchors) rather than to erase strength.

## Setup
exp069's config field-for-field except:

| field | exp069 | here | why |
|---|---|---|---|
| `erase_esd_eta` | 2 | **[1.0, 1.5, 2.0]** | the swept axis; 2.0 is the control |
| dataset | 21 rows | **33 rows** (+ exp121's 12) | more data, and the 2.0 arm measures what that alone changes |
| `steps` | 600 | **300** | erasure was flat 200 → 600; the back half measured nothing |
| `save_interval` | 100 | **50** | a lower eta should erase later and more gradually — the window where erasure and motion cross is what this run is for |

Dataset is the exp117 + exp066 + exp121 merge; build it with `merge_dataset.sh` on the cluster before
submitting (the command is in the config header). All three sources were screened at
`--min-concept-max 0.10` and carry zero blank targets under the new gate.

## What to watch
- **The crossing point.** For each arm, the earliest checkpoint with concept top-1 0.00 and the motion
  score there. Success is top-1 0.00 with motion above ~0.2 (base 0.564).
- **Colorfulness** as the second degeneracy signal — exp069 ran +40% over base while frozen; an arm
  that erases without the over-saturation is the one to take to exp071's successor.
- **The 2.0 arm against exp069.** Same eta, +12 rows, half the steps. If it also freezes, the freeze
  is not a small-data artefact — which is the most likely outcome and worth having on the record.
- **The unrelated set**, which held at −30% in exp069. A lower eta should not make that worse.
- DOVER is written as 0.0 on helios; backfill with `tools/score_dover.py` after pulling. Note from
  exp069 that DOVER did *not* separate the frozen clips from healthy ones — read `motion_score`.

## Downstream
The winning arm's checkpoint replaces exp069's in the reported row (a successor to exp071), and its
eta is what exp125 uses for the church rebuild.

## Status
- [ ] `merge_dataset.sh` run on helios; row count asserted at 33.
- [ ] Submitted.
- [ ] Crossing point recorded per arm; winner chosen.
- [ ] `docs/imagenet_objects.md` updated with the eta verdict.
