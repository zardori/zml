---
status: active
concept: nudity
method: split_prompt/precompute
thread: nudity
takeaway: >
  Sweeping split_step_frac (0.2-0.8) to replace the arbitrary 0.5 default with an empirically
  chosen value, ahead of the next frame_replace_split dataset build.
---
# exp074 — split_step_frac sweep

## Why
`split_step_frac` (fraction of the denoising schedule spent in the A/B temporal split before the
tail heals the seam on shared neutral prompt C) has been `0.5` since exp059, chosen without any
sweep — it happened to work well enough to validate the method (exp060) and build a usable dataset
(exp061). `docs/split_prompt.md` #2 names both failure modes we'd expect on either side of the
optimum: split phase too short → the concept washes out entirely (heal phase overwrites it);
split phase too long → a visible hard seam (not enough steps left to heal the temporal join).
This experiment sweeps the knob directly instead of assuming 0.5 is fine.

## Setup
`zml/precompute/split_prompt_precompute.py`, method `split_prompt`, grid over
`split_step_frac: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]` (submit_job.py grid-searches any list-valued
config field automatically — one job per value, `grid_{TIMESTAMP}/run_00N/`).

- 5 rows from `prompts/split_nudity_sweep.csv`, seeds {3103, 3111, 3124, 3147, 3163} — a subset of
  exp061 run 2's 21 human-confirmed-good triples, chosen to already be known-good scenes/prompts so
  any failure in this sweep is attributable to `split_step_frac`, not prompt quality.
- `split_latent_frame: 7` fixed (exp059's base value, pre-jitter) so `split_step_frac` is the only
  varied factor.
- `skip_plain_abc: true` (new config field, added this session) — skips regenerating the A/B/C
  reference clips, which don't depend on `split_step_frac` and were already validated in
  exp059/060; each grid job now only pays for 5 "combined" generations instead of 20.
- `save_latents: false` — this is a hyperparameter probe, not a dataset build; no need to keep
  latents per split_step_frac value.

## Evaluation plan
`zml/benchmarks/nudity_report.py` gained a `grid_dir` mode (this session) that iterates a
submit_job.py grid root's `run_*/` subdirs, reads each run's concrete `split_step_frac` from its
`config.yaml`, runs the existing per-clip NudeNet report on `run_*/outputs/videos`, and aggregates
into one `sweep_summary.json` sorted by `split_step_frac` with the second-half/first-half
localization gap per value. See exp075 (`experiments/exp075_split_step_frac_sweep_report/`) — its
`grid_dir` field needs the actual `grid_{TIMESTAMP}` path, known only once this experiment is
submitted and running.

`sweep_summary.json`'s automatic `suggested_best` (smallest split_step_frac that both localizes to
the second half and keeps the first half under threshold) is a starting point, not the final answer:
it only sees per-frame nudity scores, not seam quality. **Before locking in a value, look at the
actual `run_*/outputs/videos/*_combined.mp4` clips** for the candidates near the suggested best (and
one step above/below) to rule out the hard-seam failure mode the detector can't see.

## Status
- [ ] Submitted.
- [ ] exp075 grid_dir filled in and run.
- [ ] Candidate split_step_frac visually reviewed for seam quality.
- [ ] Value picked, `frame_replace_split_precompute.py`'s default and exp061-successor config
      updated.
