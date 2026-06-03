# exp023 — unhype fire, calibrated (first run on fixed init)

Single run, **no grid** — the point is to confirm training *starts properly* before sweeping.

## What changed vs exp020–022 (which all failed, no change vs base)

1. **Hypernet init fix** (`zml/unlearn/unhype_modules.py`). The old final layer was zero-weight,
   so the output was a constant (the bias) for every `(c, s)`: `B=0` ⇒ no-op adapter (identical
   videos) and `θ(s+1)−θ(s)≡0` ⇒ removal loss trivially won by the zero trajectory. Now the
   weight is small-but-nonzero (output varies with `(c, s)`) and the `B`-emitting rows are zeroed
   so `B=0` *exactly* at init: `A·B=0` (base preserved), `∂ℒ/∂B ∝ A ≠ 0` (B grows from the
   removal gradient), and a real `s`-trajectory can form.
2. **`simulated_lr` 0.005 → 0.3** (~60×). Endpoint `≈ S·lr·‖∇ℒ‖` was `~0.03` (inert); `0.3`
   targets `O(1)`, the scale a real adapter needs. `‖∇ℒ‖` grows as the adapter bites, so this is
   a first calibration point, not a final value.

`retain_weight=0.3` (moderate); everything else as exp021.

## Success criteria — confirm training STARTS first, in this order

1. `train/predicted_step_norm` rises toward `train/target_step_norm` (hypernet traces the
   trajectory instead of predicting zero).
2. `train/loss_remove` is non-degenerate (NOT pinned at ~1e-18).
3. `eval/theta_S_norm_concept` climbs to `O(1–10)` **and** differs across prompts (last time it
   was a constant 14.93 everywhere — no conditioning).
4. Only then judge `eval/concept_fire_detection_rate` drop vs collateral on related/unrelated.

## Next levers if it starts but under-erases

- Sweep `simulated_lr` ∈ {0.1, 0.3, 1.0} once startup is confirmed.
- Watch retain/unrelated quality: `loss_retain` only enforces `s`-constancy, not small adapter
  magnitude — if base quality degrades, add a `‖θ_retain(s)‖` penalty.
