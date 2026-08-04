---
status: active
concept: nudity
method: split_prompt/precompute
thread: nudity
takeaway: >
  Extending exp074's sweep past 0.8 toward the degenerate split_step_frac=1.0 (no heal phase at
  all) to find the actual quality ceiling, since human review found 0.4-0.8 trending upward with
  no sign of the predicted hard-seam failure yet.
---
# exp076 — split_step_frac sweep, high end (0.85-1.0)

## Why
exp074 swept `split_step_frac` in [0.2, 0.8] and, per human video review (see exp074's notes.md),
found 0.2/0.3 inconsistent (don't produce true nudity in all 5 cases) and 0.4-0.8 all consistently
good with a **slight upward tendency toward 0.8** — no seam artifact observed even at the top of
the range. That leaves the question genuinely open: does quality keep improving past 0.8, or does
`docs/split_prompt.md`'s predicted "too long a split phase -> visible hard seam" failure mode show
up somewhere in [0.85, 1.0]? `split_step_frac=1.0` is the fully degenerate case — the entire
schedule stays in the A/B split, the shared neutral prompt C never runs, so there is no seam-healing
at all. That's the natural upper bound to test against.

## Setup
Identical construction to exp074: `zml/precompute/split_prompt_precompute.py`, grid over
`split_step_frac: [0.85, 0.9, 0.95, 1.0]`, same 5 rows/seeds from `prompts/split_nudity_sweep.csv`,
`split_latent_frame: 7` fixed, `skip_plain_abc: true`, `save_latents: false`. `0.8` itself is not
re-run (already have it from exp074 `run_007`) — read this sweep's `0.85` onward against exp074's
`0.8` result for continuity.

## Evaluation plan
Same as exp074: once the grid finishes, run `scripts/benchmark.py` locally against a `nudity_report.py`
`grid_dir` config pointed at this experiment's `grid_{TIMESTAMP}` (no need for a separate cluster
job, it's CPU-only and fast) — but per exp074's own finding, **the NudeNet metric alone is not
trustworthy near a "does this look like real nudity" call**: the run_002/0.3 case scored confidently
"localized" by the metric while human review found it unreliable. Treat any automated read here as
a first pass only; the real verdict needs the same visual review exp074 got, especially watching
for the hard-seam artifact at 0.95/1.0 that the detector has no way to see at all.

## Status
- [ ] Submitted.
- [ ] Aggregate report run (locally, mirroring exp074/exp075).
- [ ] Visually reviewed by the user, especially for seam artifacts at the high end.
- [ ] Final `split_step_frac` value picked across exp074+exp076's combined range; update
      `frame_replace_split_precompute.py`'s default and the next dataset-build config.
