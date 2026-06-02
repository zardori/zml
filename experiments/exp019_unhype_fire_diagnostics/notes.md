# exp019 — unhype diagnostics (confirm the exp016 failure mode)

## Goal

`exp016_unhype_fire` produced control/fire generations indistinguishable from the base
model. This is a fast (100-step) re-run of that exact regime, but with the new
instrumentation added to `zml/unlearn/unhype.py`, to *measure* which failure mode dominates
before committing to the fixes in `exp018`.

Identical to exp016 (short OOD target/mapping prompts, `lora_rank=1`, `num_unlearning_steps=50`,
`simulated_lr=1e-3`, equal loss weights) except: short prompts kept in a dedicated file
(`prompts/cogvideox_fire_unhype_short.csv`), `steps=100`, `save_interval=50`.

## What to read (wandb / mlflow)

New metrics added to the training loop:
- `train/theta_s_norm`, `train/predicted_step_norm`, `train/target_step_norm`,
  `train/grad_theta_norm` — is the trajectory leaving the origin, and at what scale?
- `train/steering_norm` = `‖eps_target − eps_mapping‖` — strength of the ESD steering signal.
- `eval/theta_S_norm_concept` — mean endpoint adapter magnitude on the *actual* (long)
  control-fire eval prompts.

## Interpretation guide

- `eval/theta_S_norm_concept ≈ 0` → the hypernet emits a near-empty adapter on the long eval
  prompts it never saw → train/eval CLIP distribution shift dominates → fixed by long
  in-distribution training prompts (exp018) + capacity.
- `train/steering_norm` small → short OOD prompts give a weak T5 steering signal → fixed by
  long prompts (exp018).
- `train/target_step_norm ≪ train/predicted_step_norm`, or both tiny relative to a working
  ESD LoRA norm → trajectory scale too small → fixed by larger `simulated_lr` /
  `num_unlearning_steps` / `lora_rank` (exp018).

## Hypothesis

All three are expected to fire (they are coupled). Primary expectation:
`eval/theta_S_norm_concept ≈ 0` and small `steering_norm`.
