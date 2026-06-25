# exp046 — frame_replace denoising-redirection (noise the original fire latent)

## Hypothesis
exp043 (retention), exp044 (+ masked erase loss) and exp045 (+ x0-space erase loss) conclusively
showed that edited-latent reconstruction SFT cannot erase fire: across loss spaces, masking,
retention and timestep reweighting, preservation holds but concept `fire_detection_rate` stays
0.6–1.0 with no downward trend at inference. exp045 was the clincher — the x0-space `loss_erase`
sat flat at ~0.043 (individual low-noise samples ~0) and fire was completely untouched.

Root cause is a **train/inference distribution mismatch**. Every variant noises the *edited
fireless* latent and asks the model to reconstruct it — self-consistency on the fireless manifold,
a near-identity task. The fire prompt's actual sampling trajectory passes through *fire-bearing*
latents the training never visits, so the fire mode of `p(x0 | fire prompt)` is never redirected.
Driving the reconstruction loss down therefore does nothing to the fire the model samples.

This run fixes the input distribution: noise the **original (pre-edit) fire latent** — the kind of
state the model genuinely traverses when generating fire — and regress its velocity toward the
**edited fireless** target. The objective becomes "from a fire-bearing noised state under the fire
prompt, denoise toward fireless," directly redirecting the trajectory off the fire mode. This is
the single conceptual change vs the exp043 baseline (retention + masked velocity loss). The mask
stays on (original == edited on non-fire frames, so it focuses the loss on the frames that change),
retention is unchanged, and LR is held at the exp045 guardrail of 5e-4.

This is a decisive go/no-go for the whole frame_replace track: if redirecting the input
distribution still fails to erase fire, edited-latent SFT is the wrong mechanism and the next step
is a pivot to ESD-style negative guidance.

## Pipeline
1. **Precompute (new):** re-run the curated exp042 precompute with the updated
   `frame_replace_precompute.py` (now also saves `*_x0original.pt` and `original_latent_path`).
   The old exp042 outputs lack the original latents, so a fresh run is required.
   `./submit_job.py athena experiments/exp042_frame_replace_precompute_curated/config.yaml`
2. Fill this config's `metadata_file` / `latents_dir` with the new precompute's
   `outputs_{timestamp}` dir (replace the `outputs_TIMESTAMP` placeholders). Retention paths
   reuse exp041 as-is.
3. **Train:** `./submit_job.py athena experiments/exp046_frame_replace_redirect/config.yaml`

## What to watch
- **Erasure (the goal):** per-eval concept `fire_area_score_mean` and `fire_detection_rate` should
  finally trend **down** — the signal flat across exp043–045.
- **`loss_erase` should actually decrease.** Unlike the flat ~0.04–0.07 reconstruction curves, the
  redirection loss is non-trivial (input fire latent != fireless target), so a falling trajectory
  is the sign the model is learning the fire->fireless map. A flat curve here would mean the
  redirection is not being learned.
- **Preservation (must hold):** `unrelated`/`related` clip_score ~0.33 and colorfulness not
  cratering — keep the exp043 decoupling. The win condition is decoupled erasure (fire down,
  unrelated held) that exp043–045 never reached.
- `health.notes` should stay empty — no late divergence like exp044.
- If erasure unlocks but plateaus: next knobs are a mid-timestep bias (the fire/fireless input
  difference is largest at mid t) and a small `retention_weight` reduction — deferred to keep this
  a single-variable test.

## Results
- (pending first run)
