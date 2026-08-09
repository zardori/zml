---
status: active
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  Run 1 (0.5 split_step_frac, detector-derived mask) kept 4/30 and is discarded: 17 rows skipped as
  'no_concept', likely because prompt C drops the chain saw and half the schedule was heal phase,
  and the 4 kept masks came from thresholding noise. Run 2 rebuilds at split_step_frac 0.85 on the
  construction-derived mask, where nothing can be skipped.
---
# exp066 — split-prompt frame_replace dataset for "chain saw"

## Goal
First frame_replace training dataset for an ImageNet object class. A chain saw on a workbench is
present for the whole clip, so — exactly as with nudity — the partiality has to be manufactured by
the split-prompt sampler (`docs/split_prompt.md`) before the frame-local edit has anything to work
with.

Per triple: generate the combined chain-saw / object-free clip (`x0_original`), classify every frame
with ResNet-50, lift to a latent-frame concept mask, `edit_latent` onto interpolated donor frames
(`x0_edited`), save both. 30 triples, seeds 3201-3230.

## Setup
Same knobs as exp061, with the detector chosen by `concept: object` + `concept_target: "chain saw"`
through `zml/benchmarks/registry.py`. `concept_region: random` and `split_jitter: 2` keep the object's
temporal position decorrelated from the edit, so the trainer cannot learn "copy the object-free half".

The B prompts swap the chain saw for an object **outside the ten protocol classes** (bicycle pump,
watering can, paint can, ...) and are varied across the file — substituting one of the nine preserved
classes would corrupt PSR, and always substituting the *same* object would teach a fixed replacement.

`./submit_job.py helios experiments/exp066_split_chainsaw_dataset/config.yaml`

**Depends on exp064** for the `frame_concept_threshold` calibration (0.05). Since `543eed8` that field
is logging-only and gates nothing, but the calibration still matters for reading the logged
confidences when reviewing whether prompt A rendered the object at all.

## What to watch
- Keep rate in `metadata.json` vs `skipped.json` (`no_concept`, `insufficient_donor_frames`). exp061
  kept 20 of 29; a much lower rate points at the threshold.
- `videos/*_original.mp4` vs `*_edited.mp4` by hand. Two failure modes to look for: the splice never
  rendered the chain saw (should have been skipped), and the exp055 frozen-donor artefact where the
  edited region holds still instead of continuing the scene.
- `concept_region` first/second balance across kept entries.

## Run 1 (`outputs_20260803_233521`, helios, 1 h 15 m = 150 s/row) — discarded

**4 kept / 26 skipped of 30** (`skipped.json` holds 22; the four missing are the tail-flush bug fixed
in `34af14e`, which landed the day after this ran). Skip reasons: `no_concept` 17,
`insufficient_donor_frames` 5, and they are perfectly bimodal — every `no_concept` row has 13 donor
frames (nothing scored above 0.05) and every `insufficient_donor_frames` row has 0 (everything did).

The 17 `no_concept` rows are the informative ones, and the likeliest reading is a *sampler* problem
wearing a detector's clothes: at `split_step_frac: 0.5` half the schedule is the prompt-C heal phase,
and chain saw's C ("a wooden workbench in a cluttered garage") removes the object entirely, so every
heal step argues against the chain saw surviving in the concept half.

**This is a hypothesis, not a measurement.** Re-scoring exp074/exp076's nudity sweep for seam contrast
shows `split_step_frac` is nearly flat from 0.3 to 1.0 there (4/5 two-state at every value) — but
nudity's C keeps the subject and only leaves the clothed/naked attribute open, so it has little to
erase. The asymmetry is what makes the object case different; it has not been tested directly.
exp099 carries 0.5 vs 0.85 as one of its two axes precisely so this stops being an assumption.

Of the 4 kept, only `p1_s3202` is clean (26 consecutive frames at 0.20–0.55, edited trace flat at
~0.03). `p20_s3221` scraped in on two frames at 0.0513 against a 0.05 threshold and its edited max only
fell to 0.0405; `p4_s3205` still reads 0.0761 after editing. Three of the four have a single-donor
`donor_map` — the frozen fill replaced by `edit_latent_reflected` in `603b4c3`.

Seam contrast (`tools/check_seam_contrast.py`) — **2/4 two-state**:

| clip | sf | median Δ | max Δ | @ | seam | ratio | verdict |
|---|---|---|---|---|---|---|---|
| p1_s3202 | 7 | 12.408 | 16.07 | 11 | 24 | 1.3 | diffuse |
| p4_s3205 | 8 | 0.336 | 1.95 | 35 | 28 | 5.8 | two-state (weak) |
| p20_s3221 | 8 | 9.099 | 12.53 | 28 | 28 | 1.4 | diffuse |
| p25_s3226 | 7 | 0.470 | 17.03 | 24 | 24 | 36.2 | **two-state — the target shape** |

`p25_s3226` is the clip to keep in mind: a median frame-to-frame difference of 0.47 reads as "static",
but it holds two cleanly separated states with a 17.0 step exactly at its seam. Judging these clips by
average motion gets the answer backwards.

## Run 2 — rebuild
`split_step_frac: 0.5 -> 0.85`, `split_jitter: 2 -> 1`, `boundary_margin: 2` added; prompts, seeds and
every other knob unchanged. Since `543eed8` the concept mask comes from
`(split_latent_frame, concept_region)` and `insufficient_donor_frames` — now the only skip reason — is
geometric, so the outcome is predictable without running anything (simulated through `resolve_split` +
`build_edit_masks`):

- **30/30 kept**, 0 skips, `concept_region` 16 `first` / 14 `second`.
- Donor frames per row: min 3, max 7 — no row on the degenerate 2-frame floor. At jitter 2, five of
  these 30 seeds landed there (`first` at sf=9 leaves `13-9-2 = 2` donors, so the 11-frame concept
  block would be filled by ping-ponging two frames — barely better than the frozen fill that poisoned
  exp080). Jitter 1 removes all five at no cost in rows; raising `min_donor_frames` to 3 would have
  cost 5 rows to achieve the same thing.

Expect ~1 h 45 m: the split phase grows from 25 to 42 of 50 steps, and those steps cost two
transformer forwards each (75 → 92 forwards per clip).

## Downstream
Feeds exp069 — fill its `metadata_file` / `latents_dir` with run 2's `outputs_{timestamp}`.

## Status
- [x] Threshold calibrated from exp064 (now logging-only; it gates nothing since `543eed8`).
- [x] Submitted (run 1).
- [x] Run 1 reviewed — discarded, see above.
- [ ] Run 2 submitted.
- [ ] Run 2 verified: 30/30 kept, region ~15/15, masks match construction, two-state fraction well
      above run 1's 2/4.
