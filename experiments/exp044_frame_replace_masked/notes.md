# exp044 — frame_replace + retention, masked erase loss

## Hypothesis
exp043 added a retention anchor and it worked *for preservation*: the unrelated set held
(clip_score ~0.33, no collapse) unlike exp038/039. But **erasure did not happen** — concept
`fire_detection_rate` stayed 0.6–1.0 and `fire_area_score_mean` oscillated 0.037–0.127 with no
downward trend (0.060 → 0.095 over steps 100→1000). The run landed back in the "no erasure"
regime of exp035/036, just stabilized.

Root cause: the erase loss is a plain MSE averaged over **all 13 latent frames**, but the
median exp042 target has only ~6.5 edited (fire) frames — the other ~half are identical to the
base model's own output. Averaging those unedited frames in dilutes the erase gradient and
effectively *reinforces* fire on the fire prompt (a mini-retention term pointing the wrong way).
With `retention_weight=1.0` pulling hard, the net erase signal is too weak to move anything.

This run restricts the erase MSE to the **edited frames only**, using the `fire_latent_mask`
already stored per target in the exp042 metadata (`nonfire_frame_weight: 0.0` hard-masks the
rest). The retention branch is unchanged (full-tensor mean, weight 1.0). Masking is the only
changed variable vs exp043. Expectation: concept fire actually drops while preservation holds.

## Pipeline
Reuses exp041 (preservation) and exp042 (curated erase) precompute outputs as-is — no new
precompute. Config paths point at the same `outputs_{timestamp}` dirs as exp043.
- **Train**: `./submit_job.py athena experiments/exp044_frame_replace_masked/config.yaml`

## What to watch
- **Erasure (new):** per-eval concept `fire_area_score_mean` and `fire_detection_rate` should
  trend **down** — the signal that was flat in exp043.
- **Preservation (must hold):** `unrelated`/`related` clip_score ~0.33 and colorfulness not
  cratering — keep the exp043 decoupling.
- **Loss sanity:** masked `train/loss_erase` will read **higher** in absolute terms than exp043
  (the easy near-zero matched frames are no longer averaged in) — expected, not a regression.
  Both `loss_erase` and `loss_retain` should stay active and finite; `health` notes empty.
- If erasure unlocks but plateaus: next knobs are a mid/low timestep bias and a small
  `retention_weight` reduction — deferred to keep this a single-variable test.

## Results
Run `outputs_20260625_010847` (1000 steps). **Masking changed almost nothing, then the model
collapsed — the hypothesis was wrong.**

- **The mask was inert.** With the same seed, the masked `train/loss_erase` is step-for-step
  nearly identical to exp043's full-frame loss (0.073 vs 0.074, 0.069 vs 0.071, …) through step
  ~850. The exp042 targets have a *median 50% fire frames*, so masking drops half the frames from
  the average — yet the mean is unchanged. That algebraically forces the conclusion:
  **fire frames and non-fire frames carry essentially equal velocity-MSE.** The "unedited frames
  dilute/reinforce fire" premise is false; the fireless edit is simply invisible in the velocity
  objective (it's a vanishing fraction of the v-target at high-noise timesteps).
- **No erasure for 85% of the run, then full collapse.** Concept `fire_detection_rate` stayed
  0.6–1.0 through step 800 (like exp043). Then training diverged — `loss_erase` 0.072→0.326,
  `loss_retain` 0.116→0.334 climbing — and at step 1000 *everything* cratered: concept clip 0.18,
  unrelated clip 0.18, colorfulness 15/25. Fire hits 0 only because the whole model collapsed (the
  exp038/039 failure mode, delayed). `health.notes`: "loss not decreasing — erasure may be
  stalled."

Diagnosis: masking which frames you average over cannot help when every frame carries the same
near-zero fire signal; concentrating the gradient on fewer frames just raised its norm enough to
destabilize training at LR 1e-3. The real fix is to change *what space the loss lives in* so the
edit stops being buried → exp045 reparameterizes the erase loss into x0-space (with LR halved to
5e-4 as a divergence guardrail).
