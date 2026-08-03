---
status: ready
concept: nudity
method: benchmark
thread: nudity
takeaway: >
  Aggregate NudeNet report over exp074's split_step_frac grid. exp074 submitted (helios,
  grid_20260803_133224); config.yaml's grid_dir filled in. Ready to submit once exp074's 7 runs
  finish generating.
---
# exp075 — split_step_frac sweep report

## Goal
Turn exp074's grid of `combined` clips (one `run_00N/` per `split_step_frac` value) into a single
sorted table: per-value first-half/second-half NudeNet max confidence and the localization gap,
plus a starting-point `suggested_best`. Uses `zml/benchmarks/nudity_report.py`'s new `grid_dir`
mode (added alongside exp074 this session) instead of one report job per grid combination.

## Status
- [x] `config.yaml`'s `grid_dir` filled in with exp074's actual `grid_{TIMESTAMP}` path
      (`grid_20260803_133224`, helios).
- [ ] Submitted (wait for exp074's 7 runs to finish generating first).
- [ ] `sweep_summary.json` read; candidates near `suggested_best` visually checked (see exp074's
      notes) before picking a final `split_step_frac`.
