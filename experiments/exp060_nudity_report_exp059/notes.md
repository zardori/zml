# exp060 — nudity report validating the exp059 split-prompt generation

## Goal
Queued (no interactive) per-frame nudity check over exp059's clips: (a) does NudeNet fire on our
decoded frames at all, and (b) did the temporal splice localize nudity to the second half of the
combined clips?

## Results (outputs_20260725_102820, exp059 run outputs_20260724_201654)

Per-frame NudeNet score (threshold 0.3); second half = pixel frames >= 25 (latent frames [7:]).

| clip type | n | mean max | nude frames /49 | 1st-half max | 2nd-half max |
|-----------|--:|---------:|----------------:|-------------:|-------------:|
| A (naked prompt) | 4 | 0.699 | 47.25 | 0.661 | 0.692 |
| B (clothed prompt) | 4 | 0.000 | 0.00 | 0.000 | 0.000 |
| C (neutral prompt) | 4 | 0.000 | 0.00 | 0.000 | 0.000 |
| combined (split) | 4 | 0.507 | 18.00 | 0.082 | 0.507 |

`combined_verdict`: second−first half max gap = 0.425, `localized_to_second_half = True`.

## Interpretation
1. **NudeNet works on our frames** (first-run risk cleared): strong on A (0.66–0.70), exactly 0 on
   B/C. Clean detection, zero false positives on clothed/neutral — the detector is trustworthy.
2. **The splice is confirmed, quantitatively.** Combined clips are clean in the first half
   (max 0.082, below threshold → ~0 nude frames) and clearly nude in the second half (max 0.507,
   ~18/24 frames). Exactly the partial structure frame_replace needs: clothed donors early, concept
   late, in one coherent clip.
3. Combined 2nd-half strength (0.507) is below A (0.692) — the neutral-prompt coherence tail softens
   the concept a little. Well above threshold, so not a problem; raise `split_step_frac` if we want
   more explicit concept frames.

## Consequence for the dataset builder (Step 2)
The concept is a **terminal** block (second half, nothing clean after it), so `edit_latent`'s
interpolation can't bracket it → it would fall back to a one-sided copy → the exp055/057 freeze.
Two motion-safe fixes, both supported by this data:
- **#2 concept-in-the-middle**: regenerate with prompt A on middle latent frames, B on both ends, so
  clothed donors bracket the concept and interpolation applies. One coherent clip, minimal edit.
- **#3 paired same-seed** (already generated): A is nude in ~47/49 frames, B is perfectly clean, same
  seed → a frame-aligned clothed donor for every nude frame, no freeze. Simplest, full motion, but a
  larger (whole-clip) edit.

## Status
- [x] Submitted + results pulled.
- [x] Detector validated, splice localization confirmed.
