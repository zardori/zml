# exp028 — UnHype distillation control

## Why
Across exp024–exp027 the hypernet never erased fire (`concept.fire_detection_rate` = 1.0).
The root signal `loss_remove_direction = 1 − cos(θ_{s+1}−θ_s, ESD target)` is pinned ~1
(orthogonal) in every run — `predicted_step = θ_{s+1} − θ_s` is a tiny, noisy difference of
two near-identical hypernet evals matched to a non-stationary, high-variance online ESD
target. exp026 (guidance 1→3) and exp027 (K=8 timestep averaging) did not move it.

Before spending more GPU-hours on the online signal, this run **isolates the apply/eval path**
from the learning signal. It is *not* a capacity test — a single static target is trivially
representable by the output-layer bias. Its value is to localize the failure.

## What
Distill the hypernet endpoint `H(c, S)` onto `θ*`, the flat LoRA vector loaded from exp006's
direct-LoRA ESD adapter (`cogvideox_erasure_lora_step1000`), which *did* erase fire. Loss is a
static `MSE(H(c, S), θ*)` (`target_mode: distill`) — no diffusion forward in the training loop;
only the periodic eval uses the GPU. Hypernet built at rank/alpha 8 to match the adapter layout
and PEFT's `alpha/rank` scaling, so weights transfer verbatim.

Code: `load_peft_adapter_as_flat` (`zml/unlearn/unhype_modules.py`) builds `θ*` in the hypernet's
`decode` layout (per-module A then B, suffix-matched, shapes asserted); the distill branch lives
in `unhype.py`'s training loop. This run also picks up the step-embedding fix
(`Hypernetwork.forward` now normalizes s by `max_step`), though at the endpoint s=S it is moot.

## Hypothesis / how to read it
- `train/distill_cosine → ~1`, `train/loss_remove → ~0`: the hypernet reproduces θ*.
- **If `eval/concept_fire_detection_rate` drops toward the exp006 value** (unrelated quality
  held) → the apply/eval path is sound; the failure is 100% in the online signal → proceed to
  exp029.
- **If fire stays intact despite `distill_cosine → 1`** → wiring/scaling bug (decode layout,
  `alpha/rank` scaling, or eval conditioning at the wrong step). Fix that before any further
  online runs — it would have doomed all of exp024–exp027.

## Run log
- **2026-06-07, attempt 1 (helios):** OOM at `optimizer.step()`. At rank 8 the hypernet output
  layer `Linear(512 → 8.26M)` is ~4.2B params; AdamW (param + grad + m + v ≈ 68 GB) plus the 5B
  transformer + T5 + VAE overflows the 95 GB GH200. Fix: switched the optimizer to Adafactor
  (`optimizer: adafactor`) — factored second moment, no momentum buffer → optimizer state drops
  to ~tens of MB. Resubmit.
- **2026-06-07, attempt 2 (helios):** OOM moved to `loss_total.backward()` (the output-layer
  param gradient, ~16.9 GB). Fix: offload the diffusion stack (transformer/T5/VAE) to CPU during
  the distill loop — it's idle there (no diffusion forward) — and bring it back only for eval
  (`move_diffusion_stack` in `unhype.py`, guarded to distill). Resubmit.
- **2026-06-07, attempt 3 (helios), `outputs_20260607_193912`:** ran to completion, no OOM
  (offload worked). But the distillation **did not converge**: see Result.

## Result
**Ran, but the control failed *as a test*.** 2000 steps, Adafactor, lr 1e-3:
- `train/distill_cosine`: 0.0005 → **0.057** (target ~1.0) — `θ_s` still ~orthogonal to `θ*`.
- `train/theta_s_norm`: 21.5 → **20.6**, drifting *down*, away from `‖θ*‖ = 40.37`.
- `train/loss_remove` (MSE): 2.53e-4 → 2.37e-4 — barely moved.
- `eval/concept fire_detection_rate`: **1.0** at step 1000 and 2000 (fire intact).

Fire stayed intact only because the hypernet never emitted `θ*`, so the eval says nothing about
the apply/eval path. The optimization is the bottleneck: `F.mse_loss` mean-reduction over 8.26M
elements gives ~4e-9 per-element gradients, and Adafactor's eps-regularized step barely moves the
4.2B-param output layer (whose bias alone could represent `θ*` exactly — so it's an
optimization-speed problem, not capacity). **Follow-up: exp030** removes the confound by injecting
`θ*` directly through the decode/apply path (no training) and evaluating once.
