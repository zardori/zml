---
status: done
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  eta ablation [0.5, 1.0, 1.5] at exp080 run_002's settings, fire-era retention held fixed.
  exp080 showed erasure is real (human review: people become clothed) but costs -85% motion and
  -38% colorfulness even at its best point. eta=2 extrapolates PAST the donor, and 20 of 34 donors
  are frozen single-frame fills, so the current setting pushes beyond "freeze". exp080 run_002 is
  the eta=2.0 arm. RESULT: eta 1.5 (run_003) is the best of the three — video rate 0.0 across
  roughly steps 50-120 — but human review still ranks exp080 run_002 (eta 2.0) above it. It also
  beat every arm of exp085's identical grid on nudity anchors, which is what exposed exp079's
  composition problem. Its frame rate shows erasure coupled to degeneracy: 0.01 at step 70
  (colorfulness 18.6), 0.21 at step 120 (23.4), 0.76 at step 170 (33.4).
---
# exp086 — eta ablation (fire-era retention)

## Why
exp080 ran a learning-rate grid and could not answer its own question, because every arm failed the
same way. Measured against the base model on the same prompts and seeds (exp063: concept motion
**0.686**, unrelated **2.015**):

| lr | concept motion @200 | vs base | unrelated motion @200 |
|---|---|---|---|
| 5e-5 | 0.090 | -87% | 1.970 (unchanged) |
| 1e-4 | 0.030 | -96% | 2.100 (unchanged) |
| 2e-4 | 0.010 | -99% | 2.440 (unchanged) |
| 5e-4 | 0.160 | -77% | 2.580 (unchanged) |

Motion collapses on concept prompts in every arm while unrelated motion is untouched — a *targeted*
freeze, not global collapse. Higher LR only reaches it sooner. No value in the grid avoids it, so
the variable under test was the wrong one.

**Human review (2026-08-07) found the erasure is nonetheless real**, and described four phases as
training proceeds: (1) still nude, (2) distorted while the model is deciding, (3) clothed, (4)
nudity returns. The best point found was **run_002 (1e-4) step 120**. Against base on the same ten
prompts that costs **clip -7.1%, colorfulness -37.5%, motion -85.2%**.

So erasure works and is semantic — people put clothes on — but it arrives with a motion collapse
that does not clear, and it is *transient* rather than converged.

**The suspect is `erase_esd_eta`.** The erase target is

    target = (1 - eta) * teacher + eta * donor

where `teacher` is the frozen base model's prediction. eta=1 is plain SFT toward the donor; **eta=2
extrapolates past the donor**, away from the base. And 20 of this dataset's 34 donors are exp061
triples built three days before the reflected-fill fix — their `donor_map` is a single repeated
frame (`[7,7,7,7,7]`), i.e. the target literally is "emit a still image". `_sft_velocity_loss`'s own
docstring says eta<1 "stops the target partway ... instead of overfitting the donor", which is
precisely the mitigation this grid tests. exp055 measured the same pathology from frozen fills at
concept -84%; we see -85% to -99%.

## Setup
exp080 run_002's settings — same 34-triple dataset, same exp041 retention anchors, `learning_rate:
0.0001`, 200 steps, `eval_num_prompts: 10`, `global_seed: 42` — with `erase_esd_eta` gridded over
**[0.5, 1.0, 1.5]**, and the eval budget re-cut (concept only, `save_interval: 10`) as described
below.

**eta=2.0 is deliberately absent: exp080 run_002 *is* that arm**, identical in every field. Read it
as the fourth point rather than spending a job to reproduce it.

Pairs with **exp085**, the same grid on exp079's nudity retention anchors. exp086 vs exp085 at
matched eta isolates the retention set.

## Eval budget: concept only
No `control_unrelated_prompts`. The unrelated column has read `nudity_detection_rate: 0.00` with
`clip` pinned at 0.33 in every nudity run to date, and this grid's question is entirely about the
concept column — concept motion against base 0.686, and whether the clothed state is stable. An
omitted control set is now *skipped* rather than scored as an empty directory (`zml/unlearn/eval.py`);
a zero-filled row reads exactly like a real measurement of 0, which is the mistake DOVER's
0.0-on-aarch64 already taught us once.

The freed third of the budget buys `save_interval: 10` instead of 20. exp080's good state was a
~40-step window that 20-step checkpoints barely sampled — at 5e-4 it opened and closed almost
entirely between two of them — so temporal resolution is worth more here than a flat column. Clips
per run still drop 300 -> 200.

**What this gives up.** Unrelated colorfulness was the *only* place exp080 showed collateral damage,
and it was not small: 32.43 -> 50.17 over training at this exact learning rate (base 33.81), and
already +28% at the step-120 good spot, while `clip` stayed flat at 0.33 and would have reported
nothing. So "the method targets nudity well" is supported by the detection and clip columns but not
by that one. This is an acceptable trade for a mechanism grid, not for the run that produces a paper
row — the reported checkpoint needs the preservation columns measured on the external sets before
any collateral claim is made.

## What to watch
- **Concept motion, not detection rate.** The question is whether eta<1 recovers motion toward
  0.686. Detection rate at n=10 is noise (exp082); use it only for the coarse phase structure.
- **Whether phase 4 disappears.** If nudity returning at steps 140-200 is eta=2 overshooting, lower
  eta should give a stable clothed state instead of a window the model passes through. That matters
  more than the best-case number: a transient optimum is not a method.
- **The colorfulness trough as a checkpoint selector.** In exp080 its position was monotonic in LR
  (140/80/60/<=20) and the human-picked good spot sat ~40 steps after it. If that relationship holds
  across eta, we can select checkpoints without human review — worth having in a paper whose own
  finding is that the detector cannot be trusted.
- Per [[feedback-detector-metrics-not-ground-truth]], the phase structure was only visible to human
  review; no arm here should be called good on metrics alone.

## Known confound
**This grid does not fix the frozen donors, it only tests whether eta can work around them.** The
structural fix is rebuilding exp061's 21 triples with `edit_latent_reflected` (a precompute, no
GPU-days, already validated in exp078 and exp081). If eta<1 does not recover motion, that rebuild
is the next thing to run rather than a further eta search.

## Status
- [x] Config prepared (exp080 run_002 + eta grid); reuses exp080's merged `combined_dataset/`.
- [ ] Submitted.
- [ ] Concept motion compared against base 0.686 per arm.
- [ ] Phase-4 recurrence checked per arm.
- [ ] Compared against exp085 at matched eta (retention-set ablation).
