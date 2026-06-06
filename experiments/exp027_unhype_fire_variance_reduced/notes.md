# exp027 — unhype fire, variance-reduced removal target

Single run, **no grid**. Follow-up to exp024/025/026, which all left `train/loss_remove_direction`
pinned ~1 (cos≈0) and fire intact.

## Why the previous runs failed (recap)

We have now ruled out **every knob on the gradient-matching loss**:
- **exp024** (MSE, `simulated_lr` 30): closed the scale gap but the matched-norm MSE was trivially
  satisfiable → no-op adapter (`theta_S` frozen at init).
- **exp025** (cosine, `simulated_lr` 0.3): the adapter moved but in a non-erasure direction;
  `loss_remove_direction` oscillated ~0.94–1.0 with no downward trend.
- **exp026** (cosine, `negative_guidance_scale` 3.0, `remove_magnitude_weight` 0.1): same — confirming
  that scale changes the target's *magnitude*, not its *direction*.

## Diagnosis: per-step target variance

The removal target `target_step = −simulated_lr·∇θ loss_task` is built from a **single** sample —
one (target, mapping) pair, one trajectory step `s`, one diffusion timestep `t`, one random latent +
stochastic rollout. The hypernet's step `θ_{s+1}−θ_s` is a smooth, deterministic function of
`(clip, s)`; it cannot align to a target whose direction reshuffles every step. The **dominant
variance source is the diffusion timestep `t`** (gradient scale/direction vary wildly across the
schedule). So the cosine direction term never had a stable target to lock onto. We know the
*expected* ESD gradient is meaningful, because the same gradient erases fire when a LoRA is trained
on it directly (exp006).

## What changed vs exp025

- **`target_grad_batch_size` 1 → 8** — the single change under test. Average the removal target over
  K=8 (timestep, latent) snapshots taken from **one shared rollout** (so ~2× step cost, not 8×; the
  rollout dominates step cost). This cuts per-step target variance, especially the timestep
  component, giving the cosine term the expected ESD descent direction to align to.
- `remove_magnitude_weight` 1.0 → 0.1 (as in exp026: let the direction term dominate).
- `steps` 1000 → 800, `save_interval` 500 → 400 (eval + checkpoint at 400 and 800), `slurm_time`
  12h → 14h (the ~2× step cost). Everything else as exp025 (cosine, `simulated_lr` 0.3, `ngs` 1.0,
  `retain_weight` 0.3). **Rank stays 4** to isolate the variance change.
- Run on **helios** (GH200, faster + 96 GB) for the extra per-step cost.

## Success criteria (in order)

1. **`train/loss_remove_direction` falls below ~0.9 with a clear downward trend** — the gate that
   never moved in exp025/026. If it falls, per-step variance was the bottleneck.
2. `train/target_step_norm` is steadier step-to-step (lower variance) than exp025/026.
3. `eval/theta_S_norm_concept` keeps moving **and** gains cross-prompt spread (std ≫ 0.05).
4. `concept.fire_detection_rate` drops **while `concept.colorfulness_mean` holds** — genuine erasure,
   not the exp005 desaturation collapse — and `unrelated` stays clean.

## Watch for

- **Desaturation collapse:** a large FDR drop with `colorfulness_mean` collapsing is the exp005
  failure mode, not erasure. The new colorfulness metric (in `summary.json` eval scores) is the guard.
- If `loss_remove_direction` **still** won't fall even with the averaged target, the issue is not
  variance but the hypernet's capacity/leverage to realize the direction — next levers (deferred,
  do not bundle here): bump `lora_rank` 4→8 / `alpha`→8 (match exp006's erasing capacity), expand
  `prompts/cogvideox_fire_unhype.csv` (currently 16 pairs) toward ~100 diverse pairs for
  generalization, or revisit the trajectory formulation. If direction falls but erasure is weak,
  raise `K` further.
