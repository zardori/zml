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

## Run 2 — extended dataset (52 triples), `outputs_20260802_223148`
Dataset felt too small at 12 confirmed-good triples for a real training run, so `split_nudity.csv`
was extended: kept the 12 confirmed-good rows (seeds 3103-3129) and appended 40 new triples (seeds
3131-3170) in new settings (art/craft studios, spa/wellness rooms, cabins/lodges, etc.), same
static-camera template and heavy-coverage donor clothing as the originals. Same precompute config
(`split_latent_frame: 7`, `concept_region: random`, `split_jitter: 2`) — no yield-improving changes
this round, just more raw triples. `slurm_time` bumped 8h → 16h for the larger set.

- [x] Submitted.
- [x] Results pulled: 31/52 auto-kept. The original 12 (seeds 3103-3129) all reproduced as
  auto-kept, confirming precompute is deterministic per (prompt, seed) — no information lost by
  regenerating them. 19/40 new triples auto-kept; 20 skipped (`no_concept` /
  `insufficient_donor_frames`). **Row 51/seed 3170 (the last CSV row) is missing entirely again** —
  neither kept nor skipped, same anomaly as row 29/seed 3130 in run 1. Two-for-two now; looks like
  a real bug (precompute cut short on the final row, e.g. a SLURM time/signal edge case at job end)
  rather than a fluke — worth a closer look before the next extension.
- [x] Human review of the 19 new auto-kept clips (done in two passes): 9 approved (seeds 3133,
  3134, 3147, 3152, 3154, 3155, 3157, 3163, 3169 — CSV indices 14, 15, 28, 33, 35, 36, 38, 44, 50);
  10 more dropped as bad splices/edits (indices 13, 16, 25, 26, 37, 40, 41, 42, 43, 49). Combined
  with the 12 originals: **21 confirmed-good triples** in `metadata_human_filtered.json`, sourced
  entirely from this run's latents (`outputs_20260802_223148/latents`). Rejections recorded in
  `human_rejected.json` (18 total across both rounds, tagged with `round`).
- [x] Fill exp062's next run — done, points at `metadata_human_filtered.json` /
  `outputs_20260802_223148/latents`.

New-triple yield this round: 19/40 auto-kept (47.5%), then only 9/19 (47%) passed human review —
i.e. ~22.5% of new raw triples ended up usable. Better than the first estimate but still consistent
with run 1's overall yield problem; if extending again, worth first tuning
`split_step_frac`/`split_latent_frame` rather than just adding more raw triples at the same
conversion rate.
