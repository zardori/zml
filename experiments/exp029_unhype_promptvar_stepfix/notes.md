# exp029 — UnHype online: prompt-variance reduction + step-embedding fix

## Why
exp027 reduced only the **timestep** variance of the ESD removal target (K=8 snapshots from one
rollout) and `loss_remove_direction` barely moved (0.99 → 0.97). The other dominant variance
source is the **prompt**: CogVideoX is conditioned on long, detailed captions, so each
`(target, mapping)` pair yields a different ESD direction. This is the domain gap vs the
image-erasure papers the hypernet idea comes from, where short low-diversity prompts make the
target nearly stationary.

## What (two changes vs exp027)
1. **Prompt-variance reduction** (`target_prompt_batch_size: 4`): the removal target
   `-η∇_θ ℒ_task` is now averaged over a batch of prompt pairs per step (in addition to the K=8
   timesteps), via `online_removal_grad` in `unhype.py`. Each pair needs its own rollout, so cost
   is ~linear in the count — kept at 4.
2. **Step-embedding fix** (code, `unhype_modules.py:Hypernetwork.forward`): `max_step` was stored
   but unused; the raw step `s ∈ [0, S]` drove the sinusoid so the top frequency wrapped ~S/2π
   times. Now `s` is normalized by `max_step` into a monotonic `[0, π]` phase, so every frequency
   component is monotonic over the trajectory and `θ_{s+1}−θ_s` has a smooth, consistent direction
   instead of one that spins with `s`.

Gated on exp028 confirming the apply/eval path erases fire — if exp028 fails, fix the wiring bug
before running this.

## Hypothesis / how to read it
If prompt variance was the remaining blocker, `train/loss_remove_direction` should finally trend
below ~0.9 with a clear downward slope, and `train/theta_s_norm` should grow. This is a short
run (300 steps, eval at 150/300) to check the trend before committing to a full run. If the
direction loss still won't fall, the next levers in reserve are value-matching (regress θ_s →
θ_s + target_step instead of finite differences) and short concept prompts.

## Cost note
Per-step cost ≈ `target_prompt_batch_size` rollouts + `target_prompt_batch_size · K` ESD-grad
forwards. With 4 pairs × K=8 this is ~4× exp027's per-step cost (rollouts dominate). Reduce
`target_grad_batch_size` or `steps` if wall-time is tight.

## Result
_TBD — awaiting cluster run._
