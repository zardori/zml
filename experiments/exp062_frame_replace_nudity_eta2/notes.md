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
- [ ] exp061 done + path filled.
- [ ] Submitted.
- [ ] Analysis (erasure + shortcut check + collateral).
