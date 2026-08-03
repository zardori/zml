---
status: superseded
concept: fire
method: frame_replace
thread: frame_replace_fire
takeaway: >
  Grid over the ESD-style interpolated erase target (erase_esd_eta). eta=0.5 is too gentle to
  ever erase; eta=2 became the working regime. Evals captured concept and unrelated only, and
  DOVER was disabled.
---
# exp053 — frame_replace with ESD-style interpolated erase target (eta grid)

## Motivation

Plain frame_replace (exp046/exp048/exp051) regresses the erase branch **all the way** to the
donor (edited fireless) latent. But the specific donor is not our true target — it is one
fire-free clip out of many, and driving the loss to ~0 means memorizing that particular latent.
We want to move the fire prompt *toward* fireless and stop at a midpoint, not overfit the donor.

This borrows the ESD idea of regressing toward a linear combination of the model's current
prediction and a goal, rather than the goal alone. The new erase target is

```
target = pred_teacher - eta * (pred_teacher - v_donor)
       = (1 - eta) * pred_teacher + eta * v_donor
```

where `pred_teacher` is the frozen base model's prediction on the same noised fire latent +
fire prompt (LoRA adapter disabled, no grad), and `v_donor` is the usual frame_replace donor
target (velocity toward `x0_edited`). Implemented via the new `erase_esd_eta` config field in
`zml/unlearn/unlearn_frame_replace.py` (`_sft_velocity_loss`). Only the erase branch uses it;
the retention branch still regresses fully to its anchor latent.

- `eta = 1` → plain donor target (identical to exp051).
- `eta = 0` → no-op (target == base prediction).
- `0 < eta < 1` → partial redirection (interpolate teacher→donor, gentler).
- `eta > 1` → extrapolation *past* the donor, away from the teacher: `donor + (eta-1)*(donor - teacher)`.
  This is the ESD negative-guidance overshoot regime — pushes harder than the donor, but large eta
  drives the target off-manifold.

## Grid

Everything is copied from exp051 (constant LR 5e-4, 1000 steps, mid/high-t 400–1000,
gradient_accumulation 4, exp042 curated targets + exp041 retention). The only swept field:

| run | erase_esd_eta |
|-----|---------------|
| 001 | 0.5 |
| 002 | 2 |
| 003 | 5 |
| 004 | 7 |
| 005 | 10 |

**exp051 is the `eta = 1` reference point** — no need to re-run it inside this grid.

## What to look for

Per-checkpoint eval over the three control sets (concept fire / related / unrelated):

- **Erasure**: concept `fire_detection_rate` should still fall. Lower eta gives a gentler target,
  so erasure may be slower / plateau higher — the question is whether some eta erases enough.
- **Collateral**: related + unrelated quality (clip_score, colorfulness, DOVER) should stay
  flatter than exp051, since the target sits closer to the base prediction.
- **Loss floor**: with eta<1 the erase loss should settle above 0 at a stable midpoint rather
  than being driven toward memorizing the donor — check `summary.json` train trend.

Sweet spot hypothesis: an intermediate eta that keeps most of exp051's erasure while measurably
reducing collateral damage on the related/unrelated sets.

## Results (grid_20260717_230150)

The first grid submissions crashed instantly (all runs) with `FileNotFoundError` on
`metadata_file` — the exp042 precompute data sat under a doubled directory path, not the flat
path the config points at. After that was corrected, `grid_20260717_230150` completed.

Data caveats: evals captured only the **concept** (fire) and **unrelated** (collateral) sets — the
`related` set did not land — and **DOVER was 0.0 / disabled**, so quality is judged by CLIP +
colorfulness. The true base-model baseline (**exp052**) was never run, so original-model numbers
below are a proxy from exp038's least-trained checkpoints, not measured.

Reading guide: **C fire** ↓ = erasure (lower better); **C clip** = quality on fire prompts (want
~0.33); **U clip** = collateral on unrelated prompts (must stay ~0.33).

| eta | best concept-fire (step) | C clip there | U clip range | verdict |
|----:|:--|:--:|:--:|:--|
| 0.5 | 0.70 → rises to 1.0 | 0.34 | 0.33–0.34 | too gentle — never erases; fire rate climbs |
| **2** | **0.20 @ step 300** (0.20 @ s100) | **0.329** | 0.331–0.342 | **sweet spot** — erases while collateral holds at baseline |
| 5 | 0.10 @ s500 | 0.296 | 0.30–0.33 | erases, but C-clip sagging, U-clip starting to slip |
| 7 | 0.00 @ s100–500 | 0.21 | 0.23–0.31 | fire gone but model degrading (clip collapse) |
| 10 | 0.00 | 0.15–0.21 | 0.16–0.27 | broken — s100 output near-blank (clip 0.15, colf 0.0) |

Reference (proxy): base model ≈ C-fire 0.8–1.0, C-clip ~0.32, U-clip ~0.335. Plain frame_replace
(exp038) only reaches C-fire 0.2 around step 500 and bounces back to 0.6–0.8 — unstable erasure.

**Takeaway.** `eta = 2` is the knee of the curve: the only setting that meaningfully erases fire
(~0.8 → 0.2) while keeping unrelated CLIP pinned at baseline. Below it (0.5) nothing erases; above
it (5/7/10) fire dies by driving the target off-manifold, collapsing quality on *both* sets. Best
checkpoint is **eta=2 @ step 300** (C-fire 0.20, C-clip 0.329, U-clip 0.331). Caveat: eta=2 erasure
is not monotone — it regresses to C-fire 0.7 at steps 400–600 (batch-1 gradient noise knocking the
LoRA out of the basin), so stop ~step 300.

**Train-loss curves.** Very noisy per-step but healthy — the jitter is Monte-Carlo variance
(step loss = mean of only `gradient_accumulation_steps`=4 micro-samples, each randomizing timestep
t∈[400,1000), which target, and noise), not instability. Smoothed, `loss_erase` trends ~0.4 → ~0.29.

## Follow-ups

- **exp055** — motion audit: re-evals base vs the eta=2 step-300 LoRA with the new
  `motion_score_mean` metric, on full control sets. Also fills the base-model baseline (supersedes
  exp052's intent) and captures the missing `related` set. Motion matters here because frame_replace
  copies donor frames, which can freeze/slow the erased output — invisible to fire/clip/colorfulness.
- To fix the eta=2 steps-400–600 erasure regression: stratified-t sampling across the accumulation
  loop + higher `gradient_accumulation_steps` (both reduce the batch-1 gradient variance).

## Status

- [x] Submitted (initial grids crashed on data-path mismatch; rerun as grid_20260717_230150).
- [x] Results pulled.
- [x] Analysis — eta=2 @ step 300 is the sweet spot.
