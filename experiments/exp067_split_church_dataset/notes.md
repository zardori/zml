# exp067 — split-prompt frame_replace dataset for "church"

## Goal
The hard half of the pilot. `docs/comparison_targets.md` argues objects are frame_replace's native
regime because they are *spatially and temporally localized* — church tests that claim from the other
side: it is a scene-level class occupying most of the frame, and T2VUnlearning's worst class
(ESR-1 82.35 vs 100 on garbage truck and French horn). If frame_replace works on chain saw but not on
church, that is a useful boundary on the method rather than a failure of the protocol.

Same construction as exp066: 30 A/B/C triples (seeds 3301-3330), split-prompt clip -> per-frame
ResNet-50 classification -> latent concept mask -> interpolated donor edit -> `x0_original` +
`x0_edited`.

## Setup
`concept: object`, `concept_target: "church"`. The B prompts replace the church with a plain building
that has no steeple or bell tower (barn, farmhouse, warehouse, village hall, ...) and vary across the
file. Deliberately avoided: monastery/castle/bell-cote-like substitutes, which the classifier would
likely still call a church and which would leave the "concept-free" half not concept-free.

`./submit_job.py helios experiments/exp067_split_church_dataset/config.yaml`

**Depends on exp064** for `frame_concept_threshold` calibration, same as exp066 — and the value need
not match exp066's, since a frame-filling building and a compact tool sit at different probabilities.

## What to watch
- Keep rate, and specifically whether the *seam* is visible: the healing prompt C is a bare landscape,
  so the transition from "building present" to "building absent" is more structural here than in the
  chain-saw set. A hard seam mid-clip is a reason to raise `split_step_frac`.
- Whether the object-free half genuinely lacks a church (classify the B-side frames — the substitute
  building must not score as `church`).
- `videos/*_edited.mp4`: does removing a frame-filling building leave a coherent scene, or a smear?

## Downstream
Feeds exp070.

## Status
- [ ] Threshold calibrated from exp064.
- [ ] Submitted.
- [ ] Dataset reviewed; kept/skipped counts and seam quality recorded.
