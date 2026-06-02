# exp020–022 — unhype fire single runs (post-diagnostics)

These three single-run setups replace the 9-run `exp018` grid (deferred: not enough free
nodes). Run order of preference: **exp021 (moderate) → exp020 (gentle) → exp022 (aggressive)**.

## What changed since exp016 / exp019

The `exp019` diagnostics showed `target_step_norm = predicted_step_norm = theta_S_norm = 0`
while `steering_norm` was large (~60). Diagnosis: **dead LoRA zero-init**. The hypernet
zero-init'd its whole output → `A = B = 0` → `∇_{θ}ℒ_task ≡ 0` (since `∂ℒ/∂B ∝ A` and
`∂ℒ/∂A ∝ B`), so nothing ever trained — independent of any hyperparameter.

Fixed in `zml/unlearn/unhype_modules.py`: standard LoRA init (`A ~ kaiming`, `B = 0`) seeded
in the final-layer **bias** with a zero weight. Base model still preserved at init
(`A·B = 0`), retention still trivially satisfied (output constant in `s`), but `∂ℒ/∂B ∝ A ≠ 0`
so the removal gradient is now non-zero. Verified locally.

All three runs also adopt the earlier fixes: long in-distribution paired prompts
(`cogvideox_fire_unhype.csv`), `lora_rank=4`, `num_unlearning_steps=300`.

## The three setups (a single "aggressiveness" axis)

| exp | simulated_lr | retain_weight | intent |
|-----|--------------|---------------|--------|
| 020 gentle     | 0.002 | 1.0 | preservation-first; cleanest if fire still drops |
| 021 moderate   | 0.005 | 0.3 | balanced — recommended first run |
| 022 aggressive | 0.010 | 0.1 | strongest erasure; watch collateral damage |

`steps=2000`, `save_interval=500` (4 evals). Steps are heavy (each does a multi-call
reverse-diffusion latent-prep loop), so `slurm_time` is 24h; reduce `steps` if turnaround is
too slow.

## Success criteria

- First, confirm the bug is gone: `train/target_step_norm > 0`, `train/predicted_step_norm > 0`,
  and `eval/theta_S_norm_concept` clearly non-zero.
- Then: `eval/concept_fire_detection_rate` drops vs base on fire prompts, while `unrelated`
  fire rate and `clip`/`dover` quality on unrelated prompts stay close to base.
