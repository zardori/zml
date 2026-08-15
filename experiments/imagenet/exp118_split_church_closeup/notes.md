---
status: done
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  Biggest yield jump in the thread: 14/30 usable against exp067 run 2's 3/30. Both failures the
  rebuild targeted moved — `no-concept` 17 -> 11 and `not-split` 10 -> 5 — and the substitute rewrite
  is measurably the reason: whole-clip prompt B now peaks at p(church) 0.064 where exp067's tied the
  concept half at 0.247. One flaw survives: the 14 survivors are 10 `first` / 4 `second`, a
  positional skew exp122 is meant to rebalance. Dataset for exp070.
---
# exp118 — split-prompt church dataset on object-dominant prompts

## Goal
Same change as exp117, on the class that needs it more. See exp117's notes for the shared reasoning;
this file records what is specific to church.

## Why church is the harder half
`tools/screen_split_dataset.py` on exp067 run 2 (`outputs_20260808_235226`):

```
30 clips | pass 3 (10%) | not-split 10 | no-concept 17
```

The `no-concept` 17 is the same failure exp117 describes. The **10 `not-split`** rows are church's own
problem: the church was rendered, and the substitute building scored just as high. `p22_s3323` reads
0.2465 in the concept half and 0.2474 in the safe half — a contrast index of −0.002, a perfect
non-split. Across all 30 rows the concept half scores higher in only 14, which is a coin flip.

Two causes, both addressed:

- **The substitutes were church-shaped.** "a village hall", "a museum facade", "a mill house", "a
  manor house" — masonry buildings of similar scale, and ResNet-50's `church` class is not narrow.
  The reframed B prompts specify the *absence* of the identifying feature ("a stone barn with a long
  slate roof", "a single-storey village hall with a flat roof"), and no substitute has a tower, spire
  or bell-cote.
- **A was under-specified.** "a church" against eval's "a stone church with a tall steeple". Prompt A
  now names the feature, rotating across steeple / spire / bell tower / gothic tower with arched
  stained-glass windows the way the eval prompts do.

## The one clip that worked, and what it shows
`p27_s3328` is the only exp067 row that split correctly, and it is worth keeping in mind because it
breaks the pixel-space checker. Frames 0-24 show a church with a bell tower at the left; from frame
36 the tower is gone and the roofline is plain — a clean, correct, construction-aligned split.

`tools/check_seam_contrast.py` calls it **diffuse** (ratio 3.0). A bell tower is a small part of a
frame, so swapping it barely moves a whole-frame mean. `tools/screen_split_dataset.py` catches it
(contrast index +0.492, next row down +0.180). That gap is why the object thread screens on the
detector differential and keeps seam contrast for diagnosis only.

## What to watch
- `no-concept` should fall well below 17, and **`not-split` below 10** — the second is the
  church-specific number, and the one the substitute rewrite is aimed at.
- Whether the substitute buildings now read church-free: their confidences are the safe-half means in
  the screener's output. A safe-half mean still near the concept half means the B rewrite did not
  take, and the next move is substitutes from a different building family entirely, not more adjectives.
- Survivor `concept_region` balance. exp067's three survivors were all `first`; a screened set that
  stays one-sided teaches the positional shortcut regardless of how clean each clip is.

## Downstream
Replaces exp067 as exp070's dataset if it clears yield.

## Results (`outputs_20260815_014904`, helios, 3 h 21 m, 30/30 rows kept, 0 skipped)

```
30 clips | pass 14 (47%) | not-split 5 | no-concept 11
surviving concept_region balance: 10 first / 4 second
--keep-seeds 3301 3302 3303 3305 3306 3308 3309 3311 3312 3315 3316 3317 3323 3329
```

Against exp067 run 2: **3 → 14 pass**, `no-concept` 17 → 11, `not-split` 10 → 5. Church gained more
than chain saw because it had two failures to fix and both moved. Screened keep-list committed as
`outputs_20260815_014904_screened.json`.

**The substitute rewrite is the measured cause of the `not-split` half.** Whole-clip prompt B — a
plain generation of the substitute building — never exceeds p(church) 0.064 across all 30 rows,
where exp067's substitutes reached 0.247 and tied the concept half (`p22_s3323`: 0.2465 vs 0.2474).
Removing the tower, spire and bell-cote from every substitute is what did it, not the added detail.

Spot-checked by eye: `p14_s3315` and `p1_s3302` are clean two-state clips — steeple/spire present in
one region, a tower-less barn or farmhouse in the other, sky and field identical across the seam.

## Two things to carry forward

**The region skew is the live risk.** 10 first / 4 second among survivors. The 20 church eval prompts
are all ordinary full scenes with no object-free half, so they remain a valid shortcut test for
exp070 — but read that run's concept-set curve as the shortcut check, not as erasure, until exp122's
fresh seeds rebalance the set.

**Two degenerate clips.** `p9_s3310` is pure white (spatial std 0.0; prompt A and B produced the
identical blank clip, mean |A − B| 0.09) and `p21_s3322` is near-white (std 11.8). Both fell out on
the concept screen, so nothing was poisoned — but `p21_s3322`'s whole-clip A confidence is 0.445,
which means a whole-clip-based rescue would have admitted a blank video. One more reason the
detector differential is the right screen, and a reason to check gen2 for the same.

## Status
- [x] Submitted.
- [x] Screened; `no-concept` 11 and `not-split` 5 against exp067 run 2's 17 and 10.
- [x] Substitute buildings confirmed church-free — whole-clip B peaks at 0.064.
- [x] Survivor `concept_region` balance checked — **10 / 4, skewed**; exp122 rebalances.
