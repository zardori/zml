# exp055 — motion audit: base vs eta=2 (frame_replace), + real baseline

## Goal
Test whether frame_replace's donor-copy target makes the trained model **slow down / freeze** the
video on erased prompts — a failure mode that fire_detection_rate, CLIP, and colorfulness cannot
see (a frozen clip can still be fire-free, on-prompt, and colorful). exp053's headline result
(eta=2 @ step 300: fire 0.8→0.2, quality preserved) was scored *without* any motion metric, so this
re-eval checks it.

## What's new
Added `zml/eval/motion.py` (`VideoMotionScorer`): mean Farneback optical-flow magnitude between
consecutive frames, averaged over the clip (~0 = frozen). Wired into `zml/unlearn/eval.py::evaluate`
so every eval now logs `motion_score_mean` (per set) to metrics.json / summary.json / mlflow / wandb
alongside colorfulness. No training-side change; the metric is eval-only.

## Setup
`job_type: eval`, two-run grid over `lora_checkpoint_dir`:
- run_001 — **base model** (null): the true original-model reference (also fills the exp052 gap).
- run_002 — **eta=2 step-300 LoRA** (exp053 grid_20260717_230150/run_002): the erasure we're auditing.

Full control sets (all rows, matching exp052): 15 fire + 10 related_v2 + 15 unrelated. Same seeds
(baked into the CSVs) and 50 inference steps for both, so base vs erased is directly comparable —
and the `related` set (dropped by the training-time evals) is captured here via `include_related`.

## What to look for
- **Motion**: compare `concept` motion_score base (run_001) vs eta=2 (run_002). A large drop on the
  concept set = the method is freezing fire prompts. `related`/`unrelated` motion should stay near
  base (localized erasure); a drop there = collateral motion damage.
- **Cross-check**: eta=2 should still show fire_detection_rate well below base with clip/colorfulness
  near base (confirming the exp053 numbers on the full sets).
- If motion IS degraded → the fix is likely upstream in target construction (blend/interpolate the
  donor instead of hard-copying frozen frames), not a loss penalty that fights the target.

## Results
_TBD — awaiting submission (project owners submit)._

## Status
- [ ] Submitted.
- [ ] Results pulled.
- [ ] Analysis (motion base vs eta=2).
