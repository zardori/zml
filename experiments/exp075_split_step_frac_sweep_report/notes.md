---
status: blocked
concept: nudity
method: benchmark
thread: nudity
takeaway: >
  Aggregate NudeNet report over exp074's split_step_frac grid. Blocked on exp074's grid_TIMESTAMP
  path — fill in config.yaml's grid_dir once exp074 is submitted, then submit this.
---
# exp075 — split_step_frac sweep report

## Goal
Turn exp074's grid of `combined` clips (one `run_00N/` per `split_step_frac` value) into a single
sorted table: per-value first-half/second-half NudeNet max confidence and the localization gap,
plus a starting-point `suggested_best`. Uses `zml/benchmarks/nudity_report.py`'s new `grid_dir`
mode (added alongside exp074 this session) instead of one report job per grid combination.

## Status
- [ ] `config.yaml`'s `grid_dir` filled in with exp074's actual `grid_{TIMESTAMP}` path.
- [ ] Submitted.
- [ ] `sweep_summary.json` read; candidates near `suggested_best` visually checked (see exp074's
      notes) before picking a final `split_step_frac`.
