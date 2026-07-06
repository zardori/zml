# exp051 — frame_replace redirection: stable optimization, constant LR, mid/high-t bias

## Hypothesis
exp048 (grad accum 4 + cosine LR) removed exp046's eval oscillation but showed *no* erasure
(concept detection 0.7–1.0 at every checkpoint). The post-mortem says exp048 never actually
tested whether the stable objective erases:

- **Travel, not noise, was cut.** With gradient accumulation at the same LR, the step along
  the true gradient is unchanged — only noise shrinks. What changed the trajectory was the
  cosine decay: integrated LR over exp048's 600 steps ≈ 600 × (5e-4 + 2.5e-5)/2 ≈ **0.157**,
  vs exp046's **0.25** at its step-500 dip. exp048 ended at the distance-equivalent of exp046
  step ~315 — where exp046 also showed detection 0.8–1.0 — and its final 100 steps ran at
  lr < 6e-5, contributing almost nothing. Its flat detection is consistent with
  "undertrained", not "the stable objective cannot erase".
- **Low-t samples are wasted.** In velocity space the fire→fireless target component is
  `sqrt(1−acp_t)·Δx0`, which vanishes at low t: samples with t < ~400 are nearly pure
  noise-matching and dilute the erase gradient. Restricting the erase branch to t ∈ [400, 1000)
  concentrates the signal per optimizer step (~1.67× from dropping [0, 400)).

Meanwhile exp047 (full 15-prompt eval of exp046@500) showed the redirection objective *can*
reach detection 0.2 — the project's best — but with 3/15 near-grayscale videos, so the win was
partly quality collapse and partly 5-prompt luck.

This run gives the redirection objective its **best stable shot**: low gradient noise
(accum 4), full travel (constant 5e-4, 1000 steps → integrated LR 0.5 = 2× exp046@500), and
concentrated signal (`timestep_min: 400`). Objective and data untouched from exp046/exp048.
The retention branch shares the timestep range; acceptable since retention reconstruction is
easy at all t.

## Pipeline
No code changes — `lr_scheduler: constant`, `gradient_accumulation_steps`, and
`timestep_min/max` are existing `unlearn_frame_replace.py` config fields.

`./submit_job.py athena experiments/exp051_frame_replace_redirect_constant_midT/config.yaml`

Runtime budget: exp048 (2400 micro-steps + 6 evals) took 10 h (wandb); this run is 4000
micro-steps + 10 evals ≈ 17 h scaled linearly → `slurm_time` 24 h (~40% margin).
`summary.json` now carries a `runtime` block, so the actual duration is on disk this time.

## What to watch
- **Win condition:** two or more *consecutive* evals with concept `fire_detection_rate` ≤ 0.2
  AND concept `colorfulness_mean` ≥ ~25 (no grayscale collapse) AND unrelated clip ~0.33.
- **Go/no-go:** if detection stays ≥ 0.6 through step 1000 despite low noise, full travel and
  concentrated signal, the stable objective genuinely lacks an erasure direction → pivot to
  ESD-style negative guidance (the exp046 contingency). No more SFT-variant runs.
- **Attribution caveat:** this changes two knobs vs exp048 (schedule, t-range). If erasure
  works, a single cheap ablation (constant LR, full t-range) can attribute it afterwards.
- **Grayscale-collapse guard:** per-video concept colorfulness not hitting ~0 (exp046@500 had
  3/15 washed-out videos on the full set).
- **Preservation:** unrelated clip ~0.33, `health.notes` empty; watch `loss_retain` for drift
  now that retention also trains only on t ≥ 400.
- Any winning checkpoint gets an exp047-style full-set verification (15 fire + related_v2 +
  15 unrelated) before being trusted, compared against the exp052 base-model baseline.

## Results
- (pending run)
