---
status: ready
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  Rebuild of exp067 on object-dominant prompts, companion to exp117. exp067 run 2 screened 3/30:
  17 rows never rendered a church and 10 more had a substitute building reading as much "church" as
  the church did. The reframe fills the frame with the building, names the steeple/spire/bell tower
  the classifier keys on, and specifies every substitute to have no tower at all. Not submitted yet.
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

## Status
- [ ] Submitted.
- [ ] Screened; `no-concept` and `not-split` compared against exp067 run 2's 17 and 10.
- [ ] Substitute buildings confirmed church-free by their safe-half confidences.
- [ ] Survivor `concept_region` balance checked.
