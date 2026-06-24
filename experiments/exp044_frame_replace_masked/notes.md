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
- (pending first run)
