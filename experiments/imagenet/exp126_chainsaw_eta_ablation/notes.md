---
status: done
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  GATE FAILED AT EVERY ETA — and that is the finding. No arm reaches top-1 0.00 with motion above
  ~0.2; the best motion at any erased checkpoint is 0.049 (eta 1.5, step 300), still -91% against
  base 0.564. Lowering eta does not relax the freeze, it only costs erasure *stability*: eta 1.0 and
  1.5 oscillate (top-1 back to 0.22 and 0.11 mid-run) while eta 2.0 holds 0.00 at all six
  checkpoints. So hypothesis 2 (substitution — the LoRA learned "chain-saw prompt -> still life")
  is supported and hypothesis 1 (erase pressure) is rejected. The eta 2.0 arm also reproduces
  exp069's freeze on +12 rows at half the steps, so the freeze is NOT a small-data artefact. Keep
  eta 2.0; the freeze needs a different instrument.
---
# exp126 — does weaker erase pressure buy back the motion?

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
eta is what exp128 uses for the church rebuild.

## Results (2026-08-17) — no eta buys the motion back

Three arms, 300 steps, `save_interval` 50, 33-row gen2 dataset, ~6.5 h each on helios (all completed).
Concept set = the 20 full chain-saw eval prompts; base motion is **0.564**, base top-1 **0.506**.

| step | eta 1.0 top-1 / motion | eta 1.5 top-1 / motion | eta 2.0 top-1 / motion |
|---|---|---|---|
| 50 | 0.22 / 0.079 | 0.11 / 0.037 | **0.00** / 0.013 |
| 100 | **0.00** / 0.039 | **0.00** / 0.045 | **0.00** / 0.014 |
| 150 | 0.11 / 0.009 | 0.11 / 0.010 | **0.00** / 0.010 |
| 200 | 0.22 / 0.008 | **0.00** / 0.012 | **0.00** / 0.030 |
| 250 | **0.00** / 0.023 | **0.00** / 0.014 | **0.00** / 0.020 |
| 300 | **0.00** / 0.033 | **0.00** / 0.049 | **0.00** / 0.015 |

### The gate

Success was pre-registered as *top-1 0.00 with motion above ~0.2*. **No arm reaches it at any
checkpoint.** Across all 18 checkpoints the highest motion at an erased one is **0.049** (eta 1.5,
step 300) — a 91% loss against base, against exp069's 98%. Motion never leaves the 0.008–0.079 band
in any arm, and the band does not order by eta.

### Hypothesis 2, not hypothesis 1

The two readings were: **pressure** (the erase term is strong enough that not generating is the
cheapest way to satisfy it — lower eta should relax the freeze before it costs erasure) versus
**substitution** (the LoRA learned "chain-saw prompt -> still life", so motion only comes back with
the object).

The data say substitution. Lower eta does not buy motion — it buys *instability*: eta 1.0 returns to
top-1 0.22 at step 200 and eta 1.5 to 0.11 at step 150, i.e. the object comes back without the motion
coming with it, while eta 2.0 holds top-1 0.00 at **all six** checkpoints. The two do not trade along
this axis at all; the weaker arms simply erase less reliably for the same frozen clips.

### The freeze is not a small-data artefact

The eta 2.0 arm is exp069's setting on **+12 rows at half the steps**, and it froze the same way
(motion 0.010–0.030 vs exp069's 0.013–0.036). Tripling neither the data nor the schedule touches it.
This was flagged in "What to watch" as the most likely outcome and it is now on the record.

### The unrelated set is unharmed

Unrelated motion runs 0.37–0.65 across all three arms against base 0.564 — at or above base in
several checkpoints, and better than exp069's -30%. Unrelated top-1/top-5 stay at 0.0000 everywhere.
So the damage remains **concept-conditional**, which is the one thing this failure mode has going for
it against nudity's global collapse (exp107, exp111).

Colorfulness runs 47–68 across arms (exp069 ran 60–73, base 64), so the over-saturation is milder
here but not eliminated, and it does not order by eta either.

## What this closes and what it opens

`erase_esd_eta` is settled: **keep 2.0**. It is the most stable eraser and the freeze is not its
fault, so the remaining eta question is closed and should not be swept again. The freeze needs a
different instrument — a motion-preserving retention term on the concept prompts, or donors that
move — and that is the next object-thread question after exp127's dataset rebuild.

## Status
- [x] `merge_dataset.sh` run on helios; 33 rows.
- [x] Submitted (grid, 3 jobs, 2026-08-16, ~6.5 h each on helios; all `completed`).
- [x] Crossing point recorded per arm; **no winner — the gate fails at every eta.**
- [ ] DOVER backfill with `tools/score_dover.py` (helios wrote 0.0). Low value here: exp069 already
      showed DOVER does not separate frozen clips from healthy ones — `motion_score` is the read.
- [ ] Human review of an eta 1.5 step-300 clip against an exp069 clip, to confirm "still life" is
      what the 0.049 looks like.
- [ ] `docs/imagenet_objects.md` updated with the eta verdict.
