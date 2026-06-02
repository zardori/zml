# exp018 — unhype fire: trajectory scale + capacity + prompt fix

> **Deferred** (9-run grid, not enough free nodes). Superseded for now by the single runs
> `exp020`–`exp022`. Also note: this grid predates the dead-LoRA-init fix found in `exp019`;
> the fix (in `zml/unlearn/unhype_modules.py`) is required for *any* of these to train.

## Context

Follow-up to `exp016_unhype_fire` (no measurable unlearning) and the `exp019` diagnostics.
Addresses the coupled failure modes identified in the post-exp016 analysis:

1. **Trajectory never leaves the origin** — with `simulated_lr=1e-3` and `S=50`, the endpoint
   displacement was only `~0.05·∇L`, so the Hypernet-Fields bootstrap never built curvature
   and the hypernet collapsed to a tiny linear ramp of the origin gradient.
2. **Adapter too weak** — `lora_rank=1` plus the tiny magnitude → base-model behaviour.
3. **Train/eval CLIP distribution shift + weak steering** — training on short OOD phrases
   ("a fire") while evaluating on long control prompts; short prompts also weaken the
   `eps_target − eps_mapping` steering signal.

## Changes vs exp016

| field | exp016 | exp018 | why |
|-------|--------|--------|-----|
| `target_mapping_path` | short phrases | `cogvideox_fire_unhype.csv` (16 long paired prompts) | align target-side CLIP dist. with eval; stronger T5 steering |
| `lora_rank` / `lora_alpha` | 1 / 1.0 | 4 / 4.0 | adapter capacity |
| `num_unlearning_steps` (S) | 50 | 300 | match paper; longer trajectory |
| `simulated_lr` | 1e-3 | **swept** [1e-3, 5e-3, 1e-2] | trajectory step size / endpoint magnitude |
| `retain_weight` | 1.0 | **swept** [1.0, 0.1, 0.01] | rebalance vs the much smaller removal loss |
| `steps` | 300 | 3000 | enough iterations to trace the 300-long trajectory |
| `save_interval` | 100 | 1000 | 3 evals over the run |

Grid = `simulated_lr` × `retain_weight` = **9 runs** (auto-grid via `submit_job.py`).

## Success criteria

- On `control_concept` (fire) prompts: `eval/concept_fire_detection_rate` drops clearly vs the
  base model, while `unrelated` fire rate and `clip`/`dover` quality on unrelated prompts stay
  close to base (no collateral damage).
- `eval/theta_S_norm_concept` is now clearly non-zero (vs ≈0 in exp019), confirming the
  trajectory leaves the origin and the adapter is active on the long eval prompts.

## Open questions / next levers if this underperforms

- If removal still loses to retention across the whole `retain_weight` sweep → normalize the
  two losses to comparable scale in code rather than via weights.
- If far-`s` trajectory is still untraced → curriculum over `s` (train small-`s` first) or
  warm-start (deferred "reconsider the method" direction).
