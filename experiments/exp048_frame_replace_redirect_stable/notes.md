# exp048 — frame_replace redirection, stabilized (grad accum + cosine LR decay)

## Hypothesis
exp046 proved the redirection objective works: noising the original fire latent and regressing
toward the edited fireless target hit the project's best erasure at step 500 (concept
`fire_detection_rate` 0.0, unrelated clip held at 0.33). But the run never *converged* — the
eval trajectory oscillated (1.0 at 400 → 0.0 at 500 → 0.6 at 600 → 0.8–1.0 after), and part of
the step-500 win was quality collapse (2/5 concept videos near-grayscale).

The proposed root cause is optimization noise, not the objective: batch size 1 (one erase +
one retention sample per step) with a constant 5e-4 LR and no schedule makes the LoRA wander
through the erasure basin at full step size instead of settling in it. The flat `loss_erase`
curve is expected either way — the velocity target is dominated by irreducible noise-matching,
so the reducible fire→fireless component never shows as a visible downtrend.

Fix in this run (objective and data untouched from exp046):
- **`gradient_accumulation_steps: 4`** — each optimizer step averages 4 independent
  (target, timestep, noise) samples per branch, halving gradient noise.
- **`lr_scheduler: cosine`** — 5e-4 annealed to 2.5e-5 over 600 steps, so late training makes
  small refinements instead of large jumps out of the basin.
- **`eval_num_prompts: 10`** (was 5) — detection granularity ±0.1, making the trend readable.

600 optimizer steps × 4 micro-steps = 2400 erase samples (~2.4× exp046's exposure over its
useful 0–1000 range); exp046's action happened around steps 300–600.

## Pipeline
Code: `zml/unlearn/unlearn_frame_replace.py` gained `gradient_accumulation_steps` and
`lr_scheduler` (defaults 1 / "constant" reproduce the old behavior exactly).

`./submit_job.py athena experiments/exp048_frame_replace_redirect_stable/config.yaml`
(each step does 4× the forwards; 24 h budget vs exp046's 12 h for 1000 steps + 10 evals).

## What to watch
- **Convergence, not just a dip:** concept `fire_detection_rate` should trend down and *stay*
  down across the late checkpoints (400–600) as the LR anneals — the win condition is two or
  more consecutive evals at ≤ 0.1, not a single lucky checkpoint like exp046's step 500.
- **Grayscale-collapse guard:** concept `colorfulness_mean` staying ≥ ~25 and per-video
  colorfulness not hitting ~0. If erasure converges but concept videos wash out, the objective
  is destroying rather than redirecting — next step (exp049, deferred) is a mid-timestep bias
  and/or related-prompt retention anchors.
- **Preservation:** unrelated clip ~0.33 and colorfulness steady, `health.notes` empty.
- **`train/lr`** should trace the cosine; if the run diverges late despite the decay, the
  instability is not step-size-driven and the retention/erase gradient conflict becomes the
  prime suspect.
- exp047 (full-set eval of exp046 step 500) runs in parallel; if it shows the step-500 result
  was 5-prompt luck, this run's larger eval set is the first trustworthy read on the method.

## Results
- **No erasure at any checkpoint:** concept `fire_detection_rate` 0.8 / 0.8 / 0.8 / 0.8 / 1.0
  / 0.7 across steps 100–600. The win condition (two consecutive evals ≤ 0.1) was never
  approached.
- **The stabilization mechanics worked:** loss oscillation visibly reduced vs exp046
  (loss max 0.53 vs 1.27), `train/lr` traced the cosine, preservation held perfectly
  (unrelated clip 0.333–0.338, colorfulness 47–55), `health.notes` empty. Run took **10 h**
  (per wandb; not recorded on disk at the time — `summary.json` now has a `runtime` block).
- **Post-mortem — this run never tested the hypothesis.** Gradient accumulation at the same LR
  leaves the step along the true gradient unchanged (only noise shrinks); what changed the
  trajectory was the cosine decay. Integrated LR here ≈ 600 × (5e-4 + 2.5e-5)/2 ≈ **0.157**,
  vs exp046's **0.25** at its step-500 dip — i.e. this run ended at the distance-equivalent of
  exp046 step ~315, where exp046 also showed detection 0.8–1.0, and the final 100 steps ran at
  lr < 6e-5, contributing almost nothing. The flat detection is therefore consistent with
  "undertrained", not "the stable objective cannot erase". The schedule front-loaded the decay
  before the run reached exp046's action zone (~450–600).
- **Next:** exp051 keeps grad accum 4, reverts to constant 5e-4 over 1000 steps (integrated LR
  0.5 = 2× exp046@500) and adds `timestep_min: 400` to concentrate the erase signal — the
  redirection objective's best stable shot and the go/no-go before an ESD pivot.
