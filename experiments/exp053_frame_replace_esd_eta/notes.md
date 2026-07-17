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
- `0 < eta < 1` → partial redirection; the interior we sweep here.

## Grid

Everything is copied from exp051 (constant LR 5e-4, 1000 steps, mid/high-t 400–1000,
gradient_accumulation 4, exp042 curated targets + exp041 retention). The only swept field:

| run | erase_esd_eta |
|-----|---------------|
| 001 | 0.5 |
| 002 | 0.7 |
| 003 | 0.9 |

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

## Status

- [ ] Submitted (project owners submit manually).
- [ ] Results pulled.
- [ ] Analysis.
