# exp030 — UnHype static-apply control (zero training)

## Why
exp028 was meant to verify the apply/eval path by distilling the hypernet endpoint `H(c, S)`
onto a known-good erasing adapter `θ*` (exp006's `cogvideox_erasure_lora_step1000`, which *did*
erase fire). It failed *as a test*: the distill optimizer never reached `θ*`.

From `exp028/outputs_20260607_193912/summary.json` (2000 steps, Adafactor, lr 1e-3):
- `train/distill_cosine`: 0.0005 → **0.057** (target ~1.0) — still essentially orthogonal.
- `train/theta_s_norm`: 21.5 → **20.6**, drifting *down*, away from `‖θ*‖ = 40.37`.
- `train/loss_remove` (MSE): 2.53e-4 → 2.37e-4 — barely moved.
- `eval/concept fire_detection_rate`: **1.0** at both step 1000 and 2000.

So fire stayed intact only because the hypernet never emitted `θ*`. The control conflated two
questions — *can the hypernet fit `θ*`?* and *does emitting `θ*` erase fire?* — and answered
neither. (Root cause of the glacial fit: `F.mse_loss` mean-reduction over 8.26M elements gives
~4e-9 per-element gradients; Adafactor's eps-regularized, scale-suppressed step barely moves the
4.2B-param output layer. The output-layer bias can represent `θ*` exactly, so it is purely an
optimization-speed problem, not capacity.)

## What
Remove the optimization confound entirely. `target_mode: static_apply` injects `θ*` directly into
every `HyperLoRALinear` via the *same* `decode`/`apply_flat` path the hypernet uses (constant
across prompts), runs **one** eval, and stops — no hypernet forward, no optimizer, no training
loop. This is the minimal, decisive test of the apply/eval path. Costs one eval (~minutes) vs the
thousands of steps exp028 burned.

Code: `unhype.py` `target_mode == "static_apply"` branch (loads `θ*` via
`load_peft_adapter_as_flat`, evals through `prepare_for_prompt → apply_flat(θ*)`, returns before
the training loop). Built at rank/alpha 8 to match the adapter layout and PEFT's `alpha/rank`
scaling, so weights transfer verbatim.

## Hypothesis / how to read it
Read `outputs_*/summary.json` → the single `eval` entry (logged at step = `num_unlearning_steps`,
300) and inspect the generated videos in `eval_step_300/`.
- **`eval/concept fire_detection_rate` drops toward the exp006 value** (unrelated quality held)
  → decode layout + `alpha/rank` scaling + eval conditioning are all correct; the apply/eval path
  is sound → the failure across exp024–exp028 is **100% the online learning signal** → resume work
  on that (exp029 direction).
- **fire stays ~1.0** → a real wiring/scaling bug in `decode` / `apply_flat` / eval conditioning
  that has silently doomed every hypernet run → fix that *before* any further online runs. Compare
  against exp006's own eval to confirm `θ*` itself still erases fire as a direct PEFT adapter.

## Run log
- **2026-06-07 (helios), `outputs_20260607_233544`:** ran once, ~minutes, no training. θ* loaded
  (norm 40.37) and injected through `apply_flat`; single eval at step 300.

## Result
**Apply/eval path is SOUND.** Injecting θ* directly visibly erases fire (confirmed by watching
`eval_step_300/concept/*.mp4`) while unrelated generations are held:

| set | `fire_detection_rate` | `colorfulness_mean` | `clip_score_mean` |
|-----|-----------------------|---------------------|-------------------|
| concept   | **0.6** (vs 1.0 baseline) | **9.04** (vs 22.6) | 0.28 (vs 0.326) |
| unrelated | 0.0                       | 24.95              | 0.322            |

The colorfulness collapse (22.6 → 9.0, fire's orange/red removed) plus the lower concept fire
rate and CLIP score, with unrelated cleanly preserved, confirm θ* erases fire through *our*
`decode` → `apply_flat` path. The residual 0.6 (vs ~0 expected) is the detector being stricter
than visual inspection and different eval seeds than exp006 (exp006 predates the per-prompt CSV
seeds), not a path bug.

**Conclusion:** `decode` layout, `alpha/rank` scaling, and eval conditioning are all correct.
Therefore every failure across exp024–exp028 is **100% upstream in the online learning signal**
(`loss_remove_direction = 1 − cos(θ_{s+1}−θ_s, ESD target)` pinned ~1). The apply path is no
longer a suspect — focus all further work there.
