---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  eta ablation [0.5, 1.0, 1.5, 2.0] on exp079's nudity retention anchors, at exp080 run_002's
  settings. Paired with exp086 (identical grid, fire-era anchors) so the two together isolate the
  retention set — a comparison no nudity run has ever made, since every one has reused exp041's
  fire near-misses. Not yet submitted.
---
# exp085 — eta ablation on the nudity retention set

## Why
Two things are being changed relative to exp080 run_002, and each answers a question that run left
open.

**1. The retention anchors.** Every nudity `frame_replace` run — exp062, exp073, exp077, exp080 —
anchors its retention branch on exp041: 26 prompts written to be *fire* near-misses (monarch
butterflies, autumn maples, koi ponds, foxes, poppy fields, copper stills). exp041's notes call the
set "concept-agnostic," but it is not; every prompt was chosen to sit just outside fire. At
`retention_weight: 1.0` for nudity, it anchors preserved behaviour on the model's ability to render
orange animals, which is not the surface a nudity eraser threatens. exp079 built the matched
replacement and human review kept 20 of 30. Nothing has trained against it yet, so "a
concept-matched retention set helps" is an assumption, not a result.

**2. eta.** The full argument is in exp086's notes. Briefly: the erase target is
`(1 - eta) * teacher + eta * donor`, so eta=2 extrapolates *past* the donor — and 20 of this
dataset's 34 donors are exp061 triples with a single repeated frame, built before the reflected-fill
fix. The current setting therefore pushes beyond "freeze", which is consistent with exp080's
-85% to -99% concept motion against a base of 0.686. eta<1 is the documented mitigation.

exp080's own numbers make the retention question urgent rather than cosmetic: **unrelated
colorfulness inflated badly** at higher LR (30.65 -> 52.44 at 2e-4, 33.68 -> 51.98 at 5e-4, against
a 33.81 base) while clip stayed flat at 0.33 — real collateral that only the visual statistics saw.
And `loss_retain` rose in all four arms (0.084 -> 0.11-0.12), i.e. retention degraded throughout
training on anchors that had nothing to do with the concept.

## Setup
exp080 run_002's settings — same 34-triple dataset, `learning_rate: 0.0001`, 200 steps,
`eval_num_prompts: 10`, `global_seed: 42` — with two changes (plus the re-cut eval budget below):

- `retention_metadata_file` / `retention_latents_dir` -> exp079's `metadata_human_filtered.json`
  (20 anchors) instead of exp041's (25).
- `erase_esd_eta` gridded over **[0.5, 1.0, 1.5, 2.0]**. eta=2.0 *is* included here, unlike exp086,
  because no existing run pairs it with these anchors.

**exp086 is the identical grid on exp041's anchors.** exp085 vs exp086 at matched eta is the
retention-set ablation; within each, eta is the variable.

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
- **`loss_retain` and the concept column together.** The `unrelated` column is not generated here
  (see above), so the retention set's effect has to be read from the training loss and from whether
  erasure lands at the same eta as exp086. The collateral comparison that would have used
  `unrelated` moves to the external-benchmark eval of whichever checkpoint is reported.
- **Whether the anchors fight the erase objective.** exp079 found NudeNet scores its own anchors as
  nudity — 0.844 on a red bikini across all 49 frames — so on this content the two loss terms
  disagree by construction, and a model that correctly preserves swimwear can be scored as "still
  generating nudity". Watch `loss_retain`: exp080's rose steadily on fire anchors; if it rises here
  too, the conflict is real and `retention_weight` needs revisiting.
- **Concept motion against base 0.686**, same as exp086 — whether eta<1 recovers it.
- The four-phase structure human review found in exp080 (nude -> distorted -> clothed -> nude again).
  Per [[feedback-detector-metrics-not-ground-truth]] it was invisible to the metrics until someone
  watched the clips; no arm here should be called good on numbers alone.

## Known confound
Shared with exp086: **this does not fix the frozen donors.** 20 of the 34 training targets still
encode "emit a still image". The structural fix is rebuilding exp061's 21 triples with
`edit_latent_reflected` — a precompute, already validated in exp078 and exp081. If neither grid
recovers motion, that rebuild is the next thing to run.

## Status
- [x] exp079's anchors built and human-reviewed (20/30 kept).
- [x] Config prepared; reuses exp080's merged `combined_dataset/`, so no new precompute.
- [ ] Submitted (can run alongside exp086; they share no outputs).
- [ ] Compared against exp086 at matched eta (`loss_retain` + concept column).
- [ ] `loss_retain` trend compared against exp080's rising curve.
