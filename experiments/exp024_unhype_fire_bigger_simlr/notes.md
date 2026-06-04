# exp024 — unhype fire, bigger simulated_lr (close the predicted/target scale gap)

Single run, **no grid**. Lever #1 of two follow-ups to the exp023 diagnosis.

## Why exp023 failed (recap)

Generations never changed: `eval.concept.fire_detection_rate` pinned at 1.0, CLIP scores
byte-identical across steps 500/1000/1500. The endpoint adapter `θ(S)` stayed a **no-op**.

Root cause: `loss_remove = MSE(predicted_step, target_step)` had a **120× scale mismatch**
(`summary.json` health `predicted_vs_target_ratio_recent: 120.5`):
- `predicted_step = θ(s+1)−θ(s)`, natural norm **~0.13–0.36** (from `FINAL_LAYER_WEIGHT_STD`).
- `target_step = −simulated_lr·∇_θ loss_task`, norm **~0.002** (`0.3 × ‖grad‖~0.007`), *decaying*.

MSE between near-orthogonal high-dim vectors is dominated by `‖pred‖²`, so its gradient shrinks
the predicted step toward zero → `θ(s)` collapses to constant-in-`s` → endpoint stays at `θ(0)`
→ no-op adapter. (`theta_s_norm` ~15.2 frozen is a red herring: dominated by the constant
kaiming `A`-bias, blind to whether `B` grows.)

## What changed vs exp023

- **`simulated_lr` 0.3 → 30.0** (~100×). `target_step ≈ simulated_lr·‖grad_θ‖ ≈ 30·0.007 ≈ 0.2`,
  the same order as `predicted_step`. With the gap closed, MSE should actually grow `B` along the
  erasure direction instead of just shrinking the step.
- `steps` 2000 → 1000, `slurm_time` 1d → 12h. Checkpoints at 500/1000 (`hypernet.pt`) allow
  resuming if promising.
- Everything else as exp023 (`retain_weight: 0.3`, MSE loss kept — `remove_loss_type` defaulted).

## Success criteria (in order)

1. `train/predicted_step_norm` and `train/target_step_norm` land within ~1 order of magnitude
   (health `predicted_vs_target_ratio_recent` near ~1, not 120).
2. `eval/theta_S_norm_concept` moves off the frozen 15.16 **and** gains spread across prompts
   (std ≫ 0.01 — real conditioning).
3. `concept.fire_detection_rate` drops while `unrelated` (and `related`) stay clean.

## Watch for

- Instability / divergence from the large step. As `B` bites, `‖grad_θ‖` grows, so the effective
  target keeps moving — 30 is a first calibration point, not a final value.
