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

## Status
- [ ] exp056 precompute done + path filled.
- [ ] Submitted.
- [ ] Motion eval vs exp055 baseline.
