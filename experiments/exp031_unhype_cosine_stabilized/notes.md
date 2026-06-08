# exp031 — UnHype online: per-prompt conditioning fix + stabilized cosine

## Why
exp030 proved the apply/eval path is sound (injecting a known-good `θ*` erases fire), so every
failure across exp024–exp029 is in the **online learning signal**. exp029 was the first run where
that signal showed life — and then destroyed it:

- **Steps 0–49 (promising):** `loss_remove_direction` fell from ~1.13 to **0.55** (cos sim ≈ 0.45)
  with `theta_s_norm` flat at init. The exp029 changes (prompt-batch averaging + the step-embedding
  fix) exposed an alignable ESD descent direction for the first time — success criterion #1 from
  exp024–027, which had never moved.
- **Steps 50–99 (diverged):** a magnitude feedback loop. As `θ` grows, the adapter perturbs the
  model → `loss_task` ↑ → `grad_θ` ↑ → `target_step = −η·grad_θ` ↑ → the cosine **magnitude term**
  `(‖pred‖−‖tgt‖)²` chases the exploding target → `θ` grows more. `theta_s_norm` hit 376,
  `loss_remove_magnitude` 20670, the direction term was drowned out (bounced back to ~0.96), and the
  job died at step 128 before the step-150 eval. ~11 h burned, no eval.

While reading the loop I also found a real bug: exp029 conditioned the hypernet on **only the first**
prompt (`unhype.py` `_run_online`, old line 552) while `online_removal_grad` averaged the ESD target
over **all** sampled prompts — so a prompt[0]-conditioned `predicted_step` was matched against an
average target it cannot represent, capping the achievable cosine.

## What (vs exp029)
1. **Per-prompt conditioning fix (code, `unhype.py`):** `online_removal_grad` is now a *single-pair*
   helper (timestep-averaged ESD grad for one prompt at *its own* `theta_s`). `_online_removal_step`
   loops the sampled batch, conditions the hypernet on each prompt, and **averages** the per-prompt
   removal terms. `target_prompt_batch_size` now means "per-prompt terms averaged per step" — the
   consistent meaning. (Averaging, not summing, keeps the loss scale independent of batch size so lr
   / `removal_weight` stay comparable.)
2. **Target-step norm cap** (`target_step_max_norm: 1.0`): `cap_norm` rescales `target_step` down to
   the ceiling before the removal loss, so the magnitude anchor can't chase a runaway gradient — the
   exact exp029 divergence driver. Cap ≫ window-1's healthy ~0.02–0.04, ≪ the 145 explosion.
3. **Hypernet grad-norm clip** (`max_grad_norm: 1.0`): standard divergence guard, in
   `apply_optimizer_step` before `optimizer.step()`. Logs `train/grad_norm_preclip`.
4. **Divergence abort** (`theta_divergence_threshold: 80.0`, ~5× the ~15–16 init norm): `_run_online`
   stops if mean `theta_s_norm` exceeds it — minutes, not 11 h, if it blows up again.
5. **Prompt set extended 16 → 40** pairs (`prompts/cogvideox_fire_unhype.csv`) for denser coverage of
   the fire manifold and a less-biased SGD-averaged expected ESD direction. Training pairs carry no
   seeds (that policy is for the seeded `control_*` eval CSVs only).

Kept from exp029: `simulated_lr 0.3`, `lora_rank 4`, `lora_alpha 4.0`, `target_prompt_batch_size 4`,
`target_grad_batch_size 8`, `negative_guidance_scale 1.0`, `retain_weight 0.3`, cosine loss with
`remove_magnitude_weight 0.1` (anchor kept — prevents `predicted_step` collapse), `global_seed 42`.

Run config: `steps 150`, eval at **50/100/150** (50 = where the gate dropped; 100/150 confirm the
alignment *holds* past the old danger zone). `slurm_time 16h` (the abort guard is the real
protection). Run on **helios**.

## Hypothesis / how to read it
The exp029 window-1 gate drop should **reproduce and persist**: `loss_remove_direction` trends < 0.9
through step 150 (not just steps 0–49) while `theta_s_norm` grows smoothly with no >80 spike (no
divergence). With the conditioning fix the cosine should reach *lower* than exp029's 0.55. If it
holds, the step-50/100/150 evals should show `concept.fire_detection_rate` starting to drop while
`unrelated` and `colorfulness_mean` hold (genuine erasure, not the exp005 desaturation collapse).

- **Gate holds + fire starts dropping** → the online signal works; commit to a longer run / scale up.
- **Gate holds but fire intact** → signal aligns but is too weak; raise `negative_guidance_scale`,
  `lora_rank/alpha`, or steps.
- **Gate falls then climbs again without a >80 spike** → instability is subtler than magnitude
  runaway; tighten `target_step_max_norm` / `max_grad_norm`.
- **Divergence abort fires** → lower the cap / clip / `simulated_lr`.
- **Gate never falls** → the conditioning fix changed the dynamics; reconsider, and the deferred
  value-matching loss (regress `θ_{s+1} → (θ_s + target_step).detach()` with sum-reduction) becomes
  the next lever.

## Watch for
- `train/grad_norm_preclip` pinned at the clip ceiling every step → clip too tight (raise it) or lr
  too high.
- Desaturation collapse: large FDR drop with `colorfulness_mean` collapsing is the exp005 failure
  mode, not erasure.

## Result
_TBD — awaiting cluster run._
