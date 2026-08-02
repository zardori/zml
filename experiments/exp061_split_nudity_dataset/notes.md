---
status: done
concept: nudity
method: frame_replace_split/precompute
thread: nudity
takeaway: >
  First nudity frame_replace dataset, 30 triples (prompts/split_nudity.csv, seeds 3101-3130).
  20/29 auto-kept; human review left 12 confirmed-good triples in metadata_human_filtered.json.
---
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

## Status (run 1, `outputs_20260726_172848`)
- [x] Submitted.
- [x] Results pulled: 20/29 kept (row 29/seed 3130 never appeared in `metadata.json` or
  `skipped.json` — precompute likely cut short on the last row, unresolved). 9 skipped
  (`no_concept` / `insufficient_donor_frames`).
- [x] Human review (manual pass over the 20 kept clips): 8 more dropped as bad splices/edits
  (seeds 3101, 3104, 3105, 3110, 3112, 3117, 3119, 3128) → 12 confirmed-good triples fed exp062
  run 2 (`metadata_human_filtered.json`).

## Run 2 — extended dataset (52 triples)
Dataset felt too small at 12 confirmed-good triples for a real training run, so `split_nudity.csv`
was extended: kept the 12 confirmed-good rows (seeds 3103-3129) and appended 40 new triples (seeds
3131-3170) in new settings (art/craft studios, spa/wellness rooms, cabins/lodges, etc.), same
static-camera template and heavy-coverage donor clothing as the originals. Same precompute config
(`split_latent_frame: 7`, `concept_region: random`, `split_jitter: 2`) — no yield-improving changes
this round, just more raw triples. `slurm_time` bumped 8h → 16h for the larger set.

Since generation is deterministic per (prompt, seed), the 12 already-confirmed rows should
reproduce their prior good result when this reruns — no information lost, just some recomputation.

- [ ] Submitted (rerun on the 52-row extended CSV).
- [ ] Results pulled (keep rate on the 40 new triples; human review of the new ones only).
- [ ] Fill exp062's next run with the new `outputs_{timestamp}`.
