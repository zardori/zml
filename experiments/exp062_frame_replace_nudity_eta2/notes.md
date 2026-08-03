---
status: active
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  The pilot deciding whether split-prompt -> frame_replace transfers to nudity, and whether the
  positional shortcut is gone. Run 1: concept detection 0.1 -> 0.6 (step 500) -> 0.4 (step 600),
  unrelated held. Needs exp063's base reference to be interpretable.
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

- [ ] Submitted (run 2, filtered 12-triple dataset).
- [ ] Analysis (compare vs. run 1: does dropping the bad splices change erasure/collateral, or was
  20→12 just fewer gradient-step-worth of variety for the same result).
