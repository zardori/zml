---
status: done
concept: nudity
method: benchmark
thread: nudity
takeaway: >
  Ran locally instead of submitting to a cluster (CPU-only, seconds not hours) once exp074's grid
  finished. Results and the resulting split_step_frac verdict are recorded in exp074's notes.md —
  this experiment folder holds the report code path, not a separate write-up.
---
# exp075 — split_step_frac sweep report

## Goal
Turn exp074's grid of `combined` clips (one `run_00N/` per `split_step_frac` value) into a single
sorted table: per-value first-half/second-half NudeNet max confidence and the localization gap,
plus a starting-point `suggested_best`. Uses `zml/benchmarks/nudity_report.py`'s new `grid_dir`
mode (added alongside exp074 this session) instead of one report job per grid combination.

## Status
- [x] `config.yaml`'s `grid_dir` filled in (`grid_20260803_133224`).
- [x] Run — locally (`uv run python scripts/benchmark.py --config
      experiments/nudity/exp075_split_step_frac_sweep_report/config.yaml --output_dir <dir>`), not
      submitted as a cluster job; see exp074's notes.md for the numbers and verdict.
- [x] Visual seam-quality check still outstanding — tracked in exp074's Status list, not here.
