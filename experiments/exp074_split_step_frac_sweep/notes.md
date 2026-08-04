---
status: done
concept: nudity
method: split_prompt/precompute
thread: nudity
takeaway: >
  Human video review (2026-08-04) OVERRIDES the NudeNet-gap verdict below. run_001 (0.2) and
  run_002 (0.3) don't produce true nudity in ALL 5 cases — inconsistent/unreliable, not a total
  washout (0.2's near-zero metric score suggested total failure; the human read is "hit or miss").
  NudeNet's confident "localized" call on 0.3 (second-half max 0.637) doesn't match that
  inconsistency either — another instance of the detector-unreliability pattern (see
  [[feedback-detector-metrics-not-ground-truth]]). run_003-run_007 (0.4-0.8) all look similarly
  good and consistent, with a slight upward tendency toward 0.8 — the opposite direction from what
  the gap metric suggested (peak at 0.5, flat/declining after). No seam-failure ceiling was
  observed up to 0.8, the highest value tested. **Do not keep 0.5 as "confirmed best" — the real
  reliability floor is between 0.3 and 0.4, and the upper end (0.7-0.8) looks at least as good if
  not slightly better; the sweep range may need extending upward to find where quality actually
  turns over.**
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

## Results (grid_20260803_133224, all 7 runs, 5 clips each)

| split_step_frac | first_half_max | second_half_max | gap |
|---:|---:|---:|---:|
| 0.2 | 0.000 | 0.292 | 0.292 — washout, second half doesn't even clear the 0.3 detection threshold |
| 0.3 | 0.184 | 0.637 | 0.453 |
| 0.4 | 0.178 | 0.700 | 0.522 |
| **0.5** | 0.172 | **0.738** | **0.566** — best gap |
| 0.6 | 0.176 | 0.730 | 0.554 |
| 0.7 | 0.175 | 0.720 | 0.545 |
| 0.8 | 0.170 | 0.722 | 0.551 |

(Run locally via `scripts/benchmark.py` against exp075's config once exp074's grid finished — no
need to wait for a separate cluster submission for a CPU-only report. Also fixed a bug in
`nudity_report.py`'s `suggested_best` heuristic while doing this: it originally only checked
`first_half_max < threshold`, which let 0.2's total washout — both halves near zero — masquerade as
"localized". Now also requires `second_half_max` to clear `2x` the threshold.)

**Reading (superseded — kept for the record, see Human video review below):** at the time this
looked like 0.4-0.8 plateauing within noise and 0.5 as a (non-decisive) peak. That reading trusted
NudeNet's per-frame score as ground truth for "does this half show nudity," which the human review
below shows was wrong specifically at 0.3.

## Human video review (2026-08-04) — the actual verdict

- **run_001 (0.2) and run_002 (0.3): don't produce true nudity in all 5 cases** — i.e. inconsistent
  / unreliable across the 5 seeds, not a uniform failure. This is a different shape of problem than
  0.2's near-zero metric score implied (which read as a clean total washout): the human read is
  "hit or miss," not "never." NudeNet's confident "localized" call on 0.3 (second-half max 0.637,
  comfortably above the automated threshold) doesn't reflect that inconsistency at all — another
  concrete instance of [[feedback-detector-metrics-not-ground-truth]]: a single mean/max score
  across 5 clips hides exactly this kind of per-seed unreliability. Worth checking per-clip (not
  just per-run-mean) `frame_confidences` in `nudity_report_run_002.json` next time, to see if the
  metric agrees at the individual-clip level even if the run-level mean doesn't.
- **run_003 through run_007 (0.4-0.8): all look similarly good and consistent, with a slight
  upward tendency** toward the high end. No seam artifact reported even at 0.8, the highest value
  swept — so `docs/split_prompt.md`'s predicted "split phase too long -> visible hard seam" failure
  mode wasn't reached within this range.

**Consequence:** the real reliability floor sits between 0.3 and 0.4 (higher than the metric
suggested), and there's no evidence of a ceiling yet — quality trends *up* toward 0.8, not down.
Picking 0.5 because it "won" the gap metric would have picked the wrong direction; the metric's
peak and the human verdict's trend point opposite ways above 0.4. Two reasonable next steps, not
yet decided between: (a) adopt a value on the higher end (0.7-0.8) as the new default since it's at
least as consistent as 0.5 and the trend favors it, or (b) extend the sweep upward (0.9, maybe
closer to 1.0) to actually find where quality turns over before committing, since "slight upward
tendency" with no observed ceiling means 0.8 might not even be the true optimum.

## Status
- [x] Submitted.
- [x] Aggregate report run (locally, not via exp075's cluster submission — see above).
- [x] Candidate clips visually reviewed by the user — see Human video review above (supersedes the
      automated `suggested_best`/gap reading).
- [ ] Value picked: NOT decided. Either extend the sweep past 0.8 to find the real ceiling, or pick
      a value in 0.7-0.8 and move on — needs a call, don't default back to 0.5.
