# exp045 — frame_replace + retention, x0-space erase loss

## Hypothesis
exp043 (retention anchor) and exp044 (retention + masked erase loss) both failed to erase
fire. exp044's diagnosis — "unedited frames dilute the gradient, so mask them out" — turned out
to be wrong: with the same seed, exp044's masked `loss_erase` was **step-for-step nearly
identical to exp043's full-frame loss** (0.073 vs 0.074, …) through step ~850, even though the
median exp042 target has 50% fire frames and masking drops half of them from the average. A mean
that doesn't move when you remove half its terms forces one conclusion: **fire frames and
non-fire frames carry essentially equal velocity-MSE.**

Root cause: the erase loss MSEs the predicted *velocity*. The fireless edit lives in `x0`
(Δx0 ≠ 0 only on fire frames), but its contribution to the v-prediction target is scaled down by
the noise schedule and is a negligible fraction of the per-frame target norm at uniformly-sampled
(mostly high-noise) timesteps. So the erase gradient is dominated by generic noise-matching and
never moves fire — which is why the whole exp035–044 family stalls. Masking, retention tuning, or
more steps cannot fix a signal that is buried in the objective itself.

This run reparameterizes the erase loss into **x0-space**: recover the predicted clean latent
`x0_pred = sqrt(acp)·x_t − sqrt(1−acp)·v_pred` and MSE it against the edited (fireless) target on
the fire frames. There the edit is the *full* supervision signal at every timestep, not a
vanishing fraction. This is the single changed variable vs the stable exp043 baseline (the mask
stays on; the retention branch stays in velocity space). LR is halved to 5e-4 as a divergence
guardrail — exp044 collapsed at 1e-3 once its gradient concentrated, and the x0-space loss has a
different gradient scale.

## Pipeline
Reuses exp041 (preservation) and exp042 (curated erase) precompute outputs as-is — no new
precompute. Config paths point at the same `outputs_{timestamp}` dirs as exp043/exp044.
- **Train**: `./submit_job.py athena experiments/exp045_frame_replace_x0loss/config.yaml`

## What to watch
- **Erasure (the goal):** per-eval concept `fire_area_score_mean` and `fire_detection_rate`
  should finally trend **down** — the signal that was flat in exp043 and only appeared via
  collapse in exp044.
- **Preservation (must hold):** `unrelated`/`related` clip_score ~0.33 and colorfulness not
  cratering — keep the exp043 decoupling. The decoupled-erasure signature (fire down, unrelated
  held) is what neither exp043 nor exp044 achieved.
- **Loss sanity:** `train/loss_erase` reads on a **different absolute scale** than exp043/044
  (x0-space, not velocity) — expected, not a regression. `train/loss_retain` is unchanged
  (velocity-space). `health.notes` should stay empty — no late divergence like exp044.
- If erasure unlocks but plateaus: next knobs are a timestep bias and a small `retention_weight`
  reduction — deferred to keep this a single-variable test.

## Results
- (pending first run)
