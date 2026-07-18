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

## Results (grid_20260718_105906, full sets)

| set | metric | base | eta=2 s300 | Δ |
|-----|--------|-----:|-----------:|---:|
| concept | fire_detection_rate | 0.93 | 0.33 | erased |
| concept | clip_score_mean | 0.340 | 0.330 | preserved |
| concept | **motion_score_mean** | **0.58** | **0.09** | **−84%** |
| related | fire_detection_rate | 0.50 | 0.40 | — |
| related | **motion_score_mean** | **1.58** | **0.90** | **−43%** |
| unrelated | fire_detection_rate | 0.07 | 0.07 | — |
| unrelated | **motion_score_mean** | **2.01** | **1.42** | **−29%** |

(clip stays ~0.33 everywhere; colorfulness actually *rises* on eta=2 — neither sees the problem.)

**Finding: eta=2 @ step300 erases fire partly by freezing the video, and the damage is GLOBAL.**
Concept motion collapses 84% (near-frozen). Some concept drop is expected (erasing fire removes its
flicker), but related (−43%) and unrelated (−29%) have no fire to remove yet still lose large amounts
of motion — proof this is a general motion-suppression side effect of the LoRA, not localized
fire-flicker loss. Every prior metric (fire rate, CLIP, colorfulness) was blind to it. The motion
metric was worth building.

Root cause: the donor-copy target. frame_replace overwrites fire frames with *repeated frozen*
donor frames, so the target teaches "hold still," and it generalizes off the fire prompts.

Also delivered here: the real **base baseline** (concept fire 0.93 — the original model makes fire on
93% of fire prompts) and the **related** set the training-time evals had dropped.

## Proposed fix — donor interpolation in target construction

The freeze is baked into the target by `edit_latent` (`zml/unlearn/frame_replace_ops.py`): each fire
latent frame is overwritten with the **single nearest** fire-free frame (`edited[:,:,i] =
latent[:,:,donor]`). A contiguous fire block therefore becomes N *identical* frozen frames, and SFT
faithfully learns "hold still" — which then bleeds globally (related −43%, unrelated −29%). Motion is
now a first-class selection metric (`zml/eval/motion.py`, in the eval harness), so the fix has to make
the target itself keep moving. Options, in priority order:

**A. Linear latent interpolation across each fire block (primary).** Instead of copying the nearest
donor, ramp between the nearest fire-free frame *before* the block (`lo`) and *after* it (`hi`):
`w = (i - lo)/(hi - lo); edited[:,:,i] = (1-w)*latent[:,:,lo] + w*latent[:,:,hi]`. This restores a
smooth trajectory across the block instead of a freeze. Edge cases: block at clip start/end has only
one side → fall back to the one-sided copy (unavoidable, few frames). Caveat: latent-space lerp is not
pixel-linear, so long blocks become a slow cross-fade rather than true motion — still far better than a
freeze, and long all-fire spans are already skipped via `min_nofire_frames`. Small change to
`edit_latent` + a precompute re-run + one training run, then re-run the exp055 motion eval.

**B. Motion-preservation regularizer (secondary, if A leaves residual global bleed).** The unrelated
−29% comes through despite the retention branch regressing to full-motion base latents, so the LoRA's
low-rank erase direction also suppresses general motion. Add a term that matches the student's
frame-to-frame latent delta to the base model's on the retention/related branch (penalize *reduced*
motion only). Belt-and-suspenders on top of A; do not add it to the erase branch (it would fight the
target).

**C. Cheap stopgap:** prefer training clips with short fire blocks (fewer consecutive frozen frames)
by tightening the precompute filter. Reduces, doesn't remove, the freeze — not a real fix.

Plan: implement A, re-precompute, retrain (reuse exp053's eta=2 / step-300 regime), re-run the
exp055 motion eval. Escalate to B only if A doesn't restore related/unrelated motion toward base.

## Status
- [x] Submitted.
- [x] Results pulled.
- [x] Analysis — eta=2 erases but causes an 84% concept / 29–43% collateral motion collapse.
