---
status: superseded
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  SUPERSEDED by exp080 (same method on the expanded 34-triple dataset). Kept as the first nudity
  frame_replace run and for its human-review finding about detector reliability.
  Run 3 (21 human-confirmed-good triples): NudeNet concept_detection_rate 0.2->0.2->0.1->0.0(step
  400)->0.0(step 500)->0.1(step 600), no collapse signature. Human video review (2026-08-04):
  partial unlearning happened but NudeNet's 0.0 overstates it — the detector isn't reliable on
  clips with partial body visibility or multiple people, and the 21-triple dataset (all
  single-person, fully-visible, static studio-style shots) doesn't cover those cases, so it likely
  doesn't generalize to them. Not "erasure solved"; dataset breadth is the suspected next fix.
---
# exp062 — frame_replace nudity erasure (pilot, eta=2, split-prompt dataset)

## Goal
First real test that frame_replace erases **nudity**, using the manufactured partial-concept dataset
(exp061) in the exp057 eta=2 regime. This is the pilot that decides whether the whole split-prompt →
frame_replace path works before scaling the dataset.

## Setup
Same erase regime as exp057 (`erase_esd_eta: 2`, original-latent input, velocity loss, mid/high-t,
constant LR 5e-4, 600 steps, rank-8, exp041 retention). Only the dataset (exp061 nudity) and
`concept: nudity` differ — the latter makes live eval score with NudeNet.

## Dependency
Fill `metadata_file` / `latents_dir` with exp061's `outputs_{timestamp}` before submitting. Submit
order: exp061 → fill path → exp062.

## Success criteria
Live eval (concept = `cogvideox_nudity.csv`, plain full-nudity prompts):
1. **Erasure**: concept `nudity_detection_rate` drops well below base.
2. **Shortcut is gone** — this is the key check. The concept prompts are *fully* nude (no clothed
   half to copy), so if the LoRA only learned the positional "copy the other half" shortcut it will
   NOT erase here. Erasure on this set = it learned the semantic thing.
3. **Collateral**: unrelated (`cogvideox_fire_control_unrelated.csv`) clip/motion/colorfulness stay
   near base.
If erasure works and collateral holds → scale `split_nudity.csv` and regenerate a bigger exp061.
If the shortcut shows (erases the partial training clips per train loss but not the full-nudity eval)
→ revisit de-biasing (more concept_region mixing, concept-in-the-middle).

## Notes
- No nudity `related` set yet; `control_related_prompts` is a required-but-unused slot (training eval
  uses include_related=False). Build a proper nudity related set (e.g. swimwear/partial) before any
  publishable comparison.
- Retention anchors are exp041 (fire-era general prompts) — fine as a quality anchor for the pilot,
  worth a nudity-specific preservation set later.

## Status
- [x] exp061 done + path filled.
- [x] Submitted (run 1, `outputs_20260801_185906`, 20-triple auto-kept dataset).
- [x] Analysis: concept `nudity_detection_rate` 0.1→0.6(step500)→0.4(step600), unrelated held at
  0.0 detection / clip ~0.33 throughout. No base-model reference yet on these exact sets —
  exp063 (base-model baseline on the same `cogvideox_nudity.csv` + unrelated sets) fills that
  gap; fold its numbers in here once pulled.

## Run 2 — human-reviewed dataset
exp061's 20 auto-kept triples got a manual pass; 8 more were dropped as bad splices/edits (seeds
3101, 3104, 3105, 3110, 3112, 3117, 3119, 3128), on top of the 9 already auto-skipped
(`no_concept` / `insufficient_donor_frames`). 12 confirmed-good triples remain
(`metadata_human_filtered.json`, seeds 3103/3107/3109/3111/3114/3116/3121/3123/3124/3125/3127/3129).
`prompts/split_nudity.csv` was pruned to match (30 → 12 rows) — row 29/seed 3130 is a separate
anomaly (never appeared in exp061's `metadata.json` or `skipped.json`, likely the precompute run
being cut short; excluded here as unreviewed, not confirmed bad).

Config now points `metadata_file` at `metadata_human_filtered.json`; everything else (hyperparams,
retention, eval sets) unchanged from run 1, so this is a clean dataset-quality A/B. Small dataset
(12 triples) — watch for overfitting/instability vs. run 1's 20.

- [x] Submitted (run 2, filtered 12-triple dataset). Step-100 checkpoint showed
  `nudity_detection_rate` down but with `motion_score_mean` ~0.03 and near-blank videos —
  generation collapse, not genuine erasure (see exp073, which zoomed into this window).

## Run 3 — 21-triple human-reviewed dataset (`outputs_20260803_123340`)
`prompts/split_nudity.csv` extended back out to 52 rows (12 original + 40 new seeds), re-reviewed:
21/52 confirmed-good total (`metadata_human_filtered.json`, updated). Same hyperparams as runs 1-2,
just the larger, re-vetted dataset.

Ran 11.49h to step 600 on athena. Per-checkpoint (concept / unrelated `concept_detection_rate`,
concept `motion_score_mean` as the collapse tell — a blank video reads near 0 motion regardless of
`nudity_detection_rate`):

| step | concept det | concept motion | concept clip_score | unrelated det | unrelated motion |
|--:|--:|--:|--:|--:|--:|
| 100 | 0.2 | 0.176 | 0.286 | 0.0 | 1.884 |
| 200 | 0.2 | 0.292 | 0.301 | 0.0 | 1.913 |
| 300 | 0.1 | 0.391 | 0.307 | 0.0 | 2.519 |
| 400 | **0.0** | 0.228 | 0.293 | 0.0 | 1.837 |
| 500 | **0.0** | 0.221 | 0.305 | 0.0 | 1.658 |
| 600 | 0.1 | 0.240 | 0.304 | 0.0 | 1.822 |

`clip_score_mean`/`colorfulness_mean` for the concept set hold ~0.29-0.31 / 42-48 at every
checkpoint — no drop-to-near-zero the way run 2's collapsed step-100 checkpoint showed. Not
collapse, and not nothing: detection goes to 0.0 at steps 400-500 while the videos still generate
normal-looking content, and unrelated stays at 0.0 detection throughout.

**Human video review (2026-08-04) — softens the verdict above.** NudeNet's 0.0 does not mean "no
longer shows nudity": watching the actual step-400/500 concept clips, some unlearning clearly
happened, but not to the point of calling it erased. Read `nudity_detection_rate` here as a lower
bound on residual nudity, not a ground-truth measurement — the detector is known to be imperfect,
particularly on frames where the body isn't fully/cleanly visible. The suspected root cause is
**dataset coverage, not detector calibration**: all 21 training triples are the same shot type
(single person, full body visible, static studio-style framing — see `prompts/split_nudity.csv`).
The failure mode the human review surfaced — residual nudity with partial body visibility and/or
multiple people — is exactly the kind of scene this dataset never trained on, so it's an
unsurprising generalization gap, not a sign the method itself is broken. Next step: extend
`split_nudity.csv` with prompts covering partial visibility (cropped framing, objects/clothing
partially occluding) and multi-person scenes before concluding anything about erasure strength.

## Status
- [ ] `nudity_related`/preservation-specific analysis (still using exp041's fire-era retention
  anchors, per the Notes section above — untouched this round).
- [ ] Extend `split_nudity.csv` with partial-visibility and multi-person triples (see human review
  note above) — current dataset diversity, not the erase regime, is the likely blocker on
  generalization.
- [ ] Decide: once the dataset covers those cases, is this regime good enough to call "done" for
  the pilot, or does it need further scaling?
