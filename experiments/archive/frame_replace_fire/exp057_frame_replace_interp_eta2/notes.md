---
status: superseded
concept: fire
method: frame_replace
thread: frame_replace_fire
takeaway: >
  eta=2 on exp056's interpolated targets. Collateral motion fixed (unrelated back to base at
  step 300) but concept motion still -83% — half the exp055 problem solved. This is the regime
  the nudity and object transfers reuse.
---
# exp057 — frame_replace eta=2 on motion-preserving (interpolated) targets

## Goal
Re-test the exp053 eta=2 / step-300 erasure, but trained on the **interpolated** targets from exp056
(fix A) instead of the frozen-copy targets from exp042. Everything else is held identical to exp053
run_002, so this isolates whether donor interpolation removes the global motion collapse exp055 found
(concept −84%, related −43%, unrelated −29% motion) while keeping the erasure (concept fire 0.93→0.33,
CLIP preserved).

## Setup
Identical regime to exp053 run_002: `erase_esd_eta: 2`, `erase_input_latent: original`, velocity loss,
mid/high-t 400–1000, constant LR 5e-4, grad-accum 4, rank-8 LoRA, exp041 retention. **Only the erase
dataset differs** (exp056 interpolated vs exp042 frozen-copy). Shortened to 600 steps since eta=2
peaked at step 300 and regressed after.

## Dependency
`metadata_file` / `latents_dir` must be filled with exp056's `outputs_{timestamp}` path before
submitting (placeholder `outputs_TIMESTAMP` in config.yaml). Submit order: exp056 precompute → fill
path → exp057 training.

## Success criterion
Re-run the exp055 motion eval (base vs this checkpoint, full sets) on the step-300 LoRA:
- **Motion restored**: related/unrelated motion back near base (was −43% / −29%); concept motion
  recovers substantially (some concept drop is expected from removing fire's own flicker).
- **Erasure retained**: concept fire still well below base (~0.33) with CLIP ~0.33.
If related/unrelated motion is still depressed, escalate to fix B (motion-preservation regularizer on
the retention branch) — see exp055 notes.

## Results (outputs_20260719_124900)

Training-time evals (10 prompts, concept + unrelated only). Base reference = exp055 full-set eval
(concept motion 0.58, unrelated motion 2.01, concept colf 47.2, clip 0.340).

| step | fire | clip | colf | concept MOT | unrelated MOT | vs base: concept/unrel MOT |
|-----:|-----:|-----:|-----:|------------:|--------------:|:--|
| 100 | 0.00 | 0.300 | 23.1 | 0.01 | 1.37 | −99% / −32% |
| 200 | 0.60 | 0.339 | 37.5 | 0.06 | 1.66 | −90% / −18% |
| **300** | **0.30** | **0.316** | **45.1** | **0.10** | **2.20** | **−83% / +10%** |
| 400 | 0.50 | 0.339 | 52.8 | 0.06 | 1.66 | −90% / −18% |
| 500 | 0.60 | 0.333 | 52.3 | 0.07 | 2.25 | −88% / +12% |
| 600 | 0.70 | 0.333 | 57.2 | 0.07 | 1.47 | −87% / −27% |

**Fix A: half worked.**
- **Collateral motion FIXED.** exp053 (frozen-copy) held unrelated motion at 1.42 (−29%). exp057
  reaches 2.20 / 2.25 at steps 300/500 — at or *above* base. The global motion bleed onto no-fire
  prompts is gone at the good checkpoints (noisy across the rest: 1.37–1.66).
- **Concept motion NOT fixed.** Still −83% to −99% (0.01–0.10 vs base 0.58); exp053 s300 was 0.09,
  exp057 s300 is 0.10 — essentially unchanged. Interpolating the donor did not stop fire prompts
  from rendering near-static.
- Nuance: base concept motion (0.58) is already far below unrelated (2.01) — fire scenes are largely
  static with the flames as the main moving element, so erasing fire *legitimately* removes much of
  that motion. Part of the −83% is expected, not pathology. Needs video inspection to separate.

### Metric caveat: concept-set clip/colorfulness are confounded by erasure
A lower **concept CLIP** is partly the *goal*: once fire is removed the video legitimately matches a
fire-containing prompt less well. **Concept colorfulness** is confounded the same way (fire is bright
orange, so erasing it legitimately desaturates). Neither is a clean quality signal. Judge quality and
collateral on the **unrelated** (and, when available, related) sets, where there is no fire to remove.

**Step 100 — revised verdict after video inspection.** Direct viewing of the step-100 concept videos
(project owner) shows fire genuinely removed and the clips looking good. That outweighs the
metric-based "collapse" reading on the *concept* set, because concept CLIP, colorfulness AND motion are
all confounded by successful erasure there (remove the flames and you remove the scene's colour and its
main moving element). Total fire removal with clean-looking video is a significant result.

Two questions the concept-video inspection does not settle:
1. **Unrelated-set collateral** (no fire to remove, so unconfounded): step 100 is worst of the run —
   motion 1.37 (−32%), clip 0.324 (lowest), colf 60.6 (+79%, most oversaturated). Needs a look at the
   unrelated videos to judge whether that is visible/acceptable damage.
2. **Stability**: fire 0.00 @ s100 → 0.60 @ s200. Step 100 is a point the model passes through, not a
   converged erased state; and at n=10 the 0.00 has a 95% CI of ~[0, 0.31]. A held erasure is a much
   stronger claim than a single-checkpoint hit.

Unrelated metrics (base: clip 0.330, motion 2.01, colf 33.8):

| step | unrel clip | unrel motion | unrel colf |
|-----:|-----------:|-------------:|-----------:|
| 100 | 0.324 (worst) | 1.37 (−32%, worst) | 60.6 (+79%, most distorted) |
| 300 | 0.337 | 2.20 (+10%) | 44.7 (least distorted) |

Step 100 is worst on all three; step 300 is best on all three.

**Two candidate checkpoints, not one.** They optimise different things:
- **step 100** — total fire removal (0.00) with visually good concept videos; worst collateral
  metrics, and not a stable state.
- **step 300** — partial erasure (0.30) but best on every unconfounded collateral metric (unrel motion
  +10% vs base, colf least distorted), matching exp053's sweet spot.

Decide between them with a full-set eval of **both** (15 fire prompts sharpens the 0.00; related +
unrelated give clean collateral) plus a look at step-100 unrelated videos.

**New collateral finding: global oversaturation.** Unrelated colorfulness is inflated on *every*
checkpoint (33.8 base → 44.7–60.6, +32% to +79%). Unrelated prompts have no fire to remove, so this is
genuine collateral distortion, not a confound. exp053 showed the same (33.8 → 51.9). Track it
alongside motion as a first-class collateral signal.

Erasure is again noisy/non-monotone (0.00→0.60→0.30→0.50→0.60→0.70, drifting worse). With n=10 eval
prompts, fire_detection_rate carries ~±0.15 binomial noise, so single-checkpoint reads are unreliable.
Train losses healthy: loss_erase 0.354→0.263, loss_retain flat ~0.08.

**Caveat:** these are 10-prompt training-time evals without the `related` set; the base reference is a
full-set eval. Deltas are indicative, not apples-to-apples.

## Next
Run the exp055-style **full-set** eval (base vs exp057 step-300, all three sets) for a rigorous verdict
on whether fix A restored motion. Inspect concept videos to judge how much of the remaining concept
motion loss is legitimate (fire gone) vs frozen output. Escalate to fix B (motion-preservation
regularizer, exp055 notes) only if it proves pathological.

## Status
- [x] exp056 precompute done + path filled (outputs_20260718_170407).
- [x] Submitted.
- [x] Analysis — collateral motion restored; concept motion unchanged; step 300 best, step 100 degenerate.
- [ ] Full-set motion eval on step-300 checkpoint.
