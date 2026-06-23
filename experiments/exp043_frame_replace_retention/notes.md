# exp043 — frame_replace + retention anchor

## Hypothesis
In exp038 (offline) and exp039 (online), fire only drops when the *whole model* collapses: the
concept fire rate falls exactly when the unrelated set craters (clip_score 0.33→0.22,
colorfulness ~48→16), and both runs flag "loss not decreasing — erasure stalled". The edited-
latent SFT signal is too weak to remove fire *selectively*, so the optimizer reduces it by
degrading everything.

This run adds the fix `exp035/notes.md` already named: an SFT **retention anchor** toward the
base model's *unedited* preservation latents. Each step now optimizes
`loss_erase + retention_weight * loss_retain`, where `loss_retain` pins the preservation prompts
to the base model's own output. The expectation: concept fire (and the new continuous
`fire_area_score_mean`) should fall *without* the unrelated set collapsing.

## Pipeline
1. **Erase precompute** (exp042): generate + fire-edit the curated partial-fire set
   (`prompts/cogvideox_partial_fire_curated.csv`: the 21 originals plus the 50 exp040 found) into
   edited-fireless target latents.
   `./submit_job.py athena experiments/exp042_frame_replace_precompute_curated/config.yaml`
2. **Preservation precompute** (exp041): generate base-model clips for
   `prompts/cogvideox_fire_preservation.csv` and save their raw latents.
   `./submit_job.py athena experiments/exp041_preservation_precompute/config.yaml`
3. Fill this config's `metadata_file` / `latents_dir` (from exp042) and
   `retention_metadata_file` / `retention_latents_dir` (from exp041) with each run's
   `outputs_{timestamp}` dir (replace the `outputs_TIMESTAMP` placeholders).
4. **Train**: `./submit_job.py athena experiments/exp043_frame_replace_retention/config.yaml`

## What to watch
- `summary.json`: `train/loss_erase` and `train/loss_retain` should both be active (retain small
  but non-zero); per-eval `fire_area_score_mean` on the concept set should trend down.
- Decoupling check: concept `fire_detection_rate` / `fire_area_score_mean` should fall while
  `unrelated` clip_score and colorfulness hold — the opposite of the exp038/039 signature.
- `retention_weight` is the key knob: too low → collapse returns; too high → fire won't erase.
  Start at 1.0, sweep later only if the single run looks promising.

## Results
- (pending first run)
