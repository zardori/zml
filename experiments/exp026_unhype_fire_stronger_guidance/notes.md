# exp026 — unhype fire, stronger negative guidance (give the target a real erasure direction)

Single run, **no grid**. Follow-up to exp024/exp025, which both left fire fully intact at step 500.

## Why exp024 and exp025 failed (recap)

At step 500 neither touched the objective: `concept.fire_detection_rate` = 1.0, concept CLIP =
unrelated CLIP for both.

- **exp024** (MSE, `simulated_lr` 30): closed the scale gap (`predicted_vs_target_ratio` 120→0.88)
  but `theta_S_norm_concept` stayed at init (15.16, std 0.014) — a **no-op adapter**. Matching norms
  made MSE trivially satisfiable (~1e-7), so there was no pressure to move off init.
- **exp025** (cosine, `simulated_lr` 0.3): the adapter *did* move (`theta_S` 15.16→17.06, std 0.055)
  → videos visibly changed — but in a direction **unrelated to erasure** ("fire rotated/restructured,"
  not less fire). The tell: `loss_remove_direction` oscillated ~0.94–1.0 over 500 steps with **no
  downward trend** → `predicted_step` never aligned with the erasure gradient.

## Diagnosis: the bottleneck is upstream of the removal loss

`target_step = −simulated_lr·∇θ loss_task` was tiny and noisy (`loss_task` ~1e-3, `grad_theta` ~1e-2,
`target_step_norm` ~2e-3). With no stable target direction, cosine had nothing consistent to align to.
Root cause is `negative_guidance_scale = 1.0` in
`eps_steered = eps_mapping − scale·(eps_concept − eps_mapping)` (`unhype.py:273`): at scale 1.0 the
steered target barely differs from the concept prediction, so the gradient it induces is weak and
directionless.

## What changed vs exp025

- **`negative_guidance_scale` 1.0 → 3.0** — the single primary lever. Triples the erasure displacement
  of the steered target → larger, more consistent `target_step` direction for the cosine term to lock
  onto. First calibration point; push toward 5 if direction still doesn't fall.
- **`remove_magnitude_weight` 1.0 → 0.1** — let the direction term dominate; stop the magnitude anchor
  squeezing `predicted_step` toward the tiny target norm while alignment is still forming.
- Everything else as exp025 (cosine loss kept, `simulated_lr` 0.3, 1000 steps, `slurm_time` 12h,
  checkpoints at 500/1000).

## Success criteria (in order)

1. **`train/loss_remove_direction` falls** below ~0.9 with a clear downward trend (cos sim rising) —
   the thing that never happened in exp025. This is the gate; if it stays pinned ~1, the guidance is
   still too weak (bump scale) or the alignment is fundamentally stuck.
2. `train/target_step_norm` and `train/grad_theta_norm` are meaningfully larger than exp025
   (stronger, less noisy erasure signal).
3. `eval/theta_S_norm_concept` keeps moving **and** gains cross-prompt spread (std ≫ 0.05).
4. `concept.fire_detection_rate` drops while `unrelated`/`related` stay clean.

## Watch for

- Quality collapse on `unrelated`/`related` from over-strong guidance — if erasure comes but
  retention breaks, lower the scale or raise `retain_weight`.
- If `loss_remove_direction` still won't fall even at higher scale, the issue is not the target
  magnitude but the hypernet's capacity/leverage (lora_rank 4) to realize that direction — revisit
  rank or the trajectory formulation next.
