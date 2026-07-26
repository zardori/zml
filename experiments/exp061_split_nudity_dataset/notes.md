# exp061 — pilot frame_replace dataset for nudity (split-prompt construction)

## Goal
First end-to-end build of a nudity frame_replace dataset using the validated split-prompt generation
(exp059/060). Per triple: generate the combined clothed/naked clip (x0_original) → NudeNet per-frame
→ concept latent mask → `edit_latent` (interpolated donors) → x0_edited. 30 triples
(`prompts/split_nudity.csv`, seeds 3101–3130).

## De-biasing (from the shortcut discussion)
`concept_region: random` + `split_jitter: 2` mix which half holds the concept and where the boundary
sits, so concept *position* is decorrelated from the edit. Without this the trainer could learn "copy
the concept-free half onto the other half" instead of removing the concept.

## What to check in the output
- `metadata.json`: how many of the 30 kept vs `skipped.json` (`no_concept` = splice didn't render
  nudity; `insufficient_donor_frames`). A high keep rate means the splice is reliable at scale.
- `videos/*_original.mp4` vs `*_edited.mp4`: the edit should remove nudity while keeping motion
  (interpolated donors, not frozen — watch for the exp055 failure since some concept blocks may be
  terminal despite the random side).
- `concept_region` distribution across kept targets (should be a first/second mix).

## Downstream
Feeds exp062 (frame_replace training, concept=nudity, eta=2). Fill exp062's metadata_file/latents_dir
with this run's `outputs_{timestamp}` once done.

## Status
- [ ] Submitted.
- [ ] Results pulled (keep rate, edit quality, fill exp062 path).
