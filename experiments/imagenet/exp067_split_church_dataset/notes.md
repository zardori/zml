---
status: superseded
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  Run 1 (detector-derived mask) kept 7/30 and was discarded. Run 2 (construction mask, 0.85) kept
  30/30 but screens at only 3/30: 17 rows never rendered a church (the same 17 as chain saw) and 10
  more had the substitute building scoring as much "church" as the church did. Two independent
  prompt failures, no sampler failure. Superseded by exp118. Also the source of the clearest
  evidence that seam contrast must not be used to select rows — it rejects p27_s3328, the one
  correct split in the batch.
---
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

`./submit_job.py helios experiments/imagenet/exp067_split_church_dataset/config.yaml`

**Depends on exp064** for `frame_concept_threshold` calibration, same as exp066 — and the value need
not match exp066's, since a frame-filling building and a compact tool sit at different probabilities.

## What to watch
- Keep rate, and specifically whether the *seam* is visible: the healing prompt C is a bare landscape,
  so the transition from "building present" to "building absent" is more structural here than in the
  chain-saw set. A hard seam mid-clip is a reason to raise `split_step_frac`.
- Whether the object-free half genuinely lacks a church (classify the B-side frames — the substitute
  building must not score as `church`).
- `videos/*_edited.mp4`: does removing a frame-filling building leave a coherent scene, or a smear?

## Run 1 (`outputs_20260803_233856`, helios, 1 h 16 m) — discarded

**7 kept / 23 skipped of 30** (`skipped.json` holds 21; the two missing are the tail-flush bug fixed
in `34af14e`, which landed the day after this ran). Skip reasons: `insufficient_donor_frames` 12,
`no_concept` 9 — and they are perfectly bimodal, every `no_concept` row having 13 donor frames and
every `insufficient_donor_frames` row having 0. The detector was not finding a church and failing; it
was reading noise and splitting it at 0.03.

Six of the seven *kept* rows have `frame_confidences` that never leave the 0.021–0.050 band, and four
have `edited_max_confidence` at or above the 0.03 threshold (0.0469, 0.0473, 0.0340, 0.0317) — by its
own detector the edit removed nothing. `p10_s3311` (`region: first`, `sf=6`) should carry the mask
`CCCCCC.......`; it got `CC....C......`. With `nonfire_frame_weight: 0.0` the erase loss is
hard-masked to that mask, so training would have applied it to the safe half.

**All seven kept rows are `concept_region: first`** despite `concept_region: random` — a
detector-selection artifact, and on its own disqualifying: the dataset would have taught the
positional shortcut `docs/split_prompt.md` §4 warns about.

Seam contrast (`tools/check_seam_contrast.py`, on the run's own MP4s) — **1/7 two-state**:

| clip | sf | median Δ | max Δ | @ | seam | ratio | verdict |
|---|---|---|---|---|---|---|---|
| p10_s3311 | 6 | 0.421 | 1.66 | 23 | 20 | 3.9 | diffuse |
| p12_s3313 | 9 | 0.391 | 1.69 | 11 | 32 | 4.3 | diffuse |
| p13_s3314 | 6 | 1.293 | 3.00 | 20 | 20 | 2.3 | diffuse |
| p16_s3317 | 9 | 17.566 | 20.91 | 22 | 32 | 1.2 | diffuse — motion smears the seam away |
| p19_s3320 | 6 | 2.163 | 2.88 | 11 | 20 | 1.3 | diffuse |
| p21_s3322 | 6 | 0.211 | 0.81 | 0 | 20 | 3.8 | **collapsed to one state** |
| p27_s3328 | 6 | 0.249 | 3.12 | 25 | 20 | 12.5 | **two-state** (also the only row the detector genuinely fired on, 0.1995) |

Note p27_s3328's median of 0.249: a clip can be almost static within each half and still be a good
target. What matters is the step at the seam, not the motion.

**Read that 1/7 with a caveat.** Seam contrast is a whole-frame mean, and church swaps one building
inside an otherwise identical village, so the step is intrinsically small — nudity, which swaps most
of the subject, scores a median seam ratio of ~14 against church's 3.8. The `max/median`
normalisation absorbs part of that but not all of it, so some of these "diffuse" rows may have split
correctly with a subtle difference. It does not rescue the dataset: the mask, region-balance and
`edited_max_confidence` failures above are independent of this metric and are each disqualifying on
their own. But use the two-state fraction to compare run 2 against run 1 on *church*, not against
nudity's numbers.

## Run 2 — rebuild
`split_step_frac: 0.5 -> 0.85`, `split_jitter: 2 -> 1`, `boundary_margin: 2` added; prompts, seeds and
every other knob unchanged.

Because the skip gate is now purely geometric and deterministic given the seeds, the outcome is
predictable without running anything (simulated through `resolve_split` + `build_edit_masks`):

- **30/30 kept**, 0 skips.
- Donor frames per row: min 3, max 7 — no row on the degenerate 2-frame floor. At jitter 2, five of
  these 30 seeds landed there (`first` at sf=9 leaves `13-9-2 = 2` donors, so the 11-frame concept
  block would be filled by ping-ponging two frames). That is what jitter 1 buys, at no cost in rows.
- **`concept_region` comes out 22 `first` / 8 `second`.** Much better than run 1's 7/7, but still a
  73/27 skew — an unlucky draw from the per-seed coin flip (p ≈ 1.6% under a fair coin), not a bug,
  and not fixable without changing the committed seeds. Worth knowing because a trainer could satisfy
  73% of the data with "erase the first half". The check that catches it is the eval-time one: the 20
  full-church eval prompts have no church-free half to copy, so a positional rule cannot score on
  them.

## Run 2 result (`outputs_20260808_235226`, helios)

30/30 kept, `skipped.json` empty, `concept_region` 22 first / 8 second (the predicted unlucky draw),
masks matching construction on every row. Then:

```
30 clips | pass 3 (10%) | not-split 10 | no-concept 17
surviving concept_region balance: 3 first / 0 second
--keep-seeds 3308 3314 3328
```

**Two independent prompt failures, neither of them the sampler:**

1. **17 rows never rendered a church** — the same 17 of 30 as chain saw, which is what makes the cause
   structural rather than per-class. Wide framing puts a small building in a large landscape.
2. **10 rows rendered a church and a church-shaped substitute.** `p22_s3323` reads 0.2465 in the
   concept half against 0.2474 in the safe half — a contrast index of −0.002, a perfect non-split.
   Across all 30 rows the concept half scores higher in only 14, a coin flip. "a village hall", "a
   museum facade", "a manor house", "a mill house" are masonry buildings of similar scale, and
   ResNet-50's `church` class is not narrow enough to tell them apart.

Note also the survivor balance: 3 first / 0 second. The full set is 22/8, but screening *concentrated*
the skew — worth remembering that a keep-list can be more biased than the build it came from.

### The clip that proves seam contrast can't select rows
`p27_s3328` is the only row that split correctly. Frames 0–24 show a church with a bell tower at the
left; from frame 36 the tower is gone and the roofline is plain — clean, correct and aligned with the
construction seam.

`tools/check_seam_contrast.py` calls it **diffuse** (ratio 3.0), because a bell tower is a small share
of a 480×720 frame and the metric is a whole-frame mean. `tools/screen_split_dataset.py` catches it at
contrast index +0.492, with the next row down at +0.180 — a wide gap. Combined with exp066's
`p13_s3214` (a textbook seam containing no chain saw at all), this is the pair of counterexamples that
moved row selection onto the detector differential. See `docs/split_prompt.md` §3.1.

## Downstream
Superseded by **exp118**, which rebuilds on object-dominant prompts with substitutes specified to have
no tower, spire or bell-cote. exp070 should point at exp118's output.

## Status
- [x] Threshold calibrated from exp064 (now logging-only; it gates nothing since `543eed8`).
- [x] Submitted (run 1).
- [x] Run 1 reviewed — discarded, see above.
- [x] Run 2 submitted and verified: 30/30 kept, region 22/8, masks match construction.
- [x] Run 2 screened — 3/30 usable, 17 no-concept + 10 not-split. Superseded by exp118.
