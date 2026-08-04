---
status: active
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  Testing whether a linear LR warmup fixes the softness/lack-of-sharpness human review found in
  exp073's clips, without losing the erasure signal. New lr_warmup_steps config field added to
  unlearn_frame_replace.py this session. Grid: [0, 30, 60] warmup steps, otherwise identical to
  exp073.
---
# exp077 — frame_replace nudity, LR warmup test

## Why
exp073's clips are genuinely erased (people really do have clothes) but human review found them
visibly soft/not sharp. None of the pipeline's per-checkpoint metrics catch this directly —
`clip_score`/`colorfulness`/`motion` are semantic/color/movement proxies, and `dover_technical_mean`/
`dover_aesthetic_mean` (the metric that would actually measure this) read `0.0` at every checkpoint
because DOVER is unavailable in this run's environment, not because quality is truly zero.

Circumstantial evidence pointed at the LR schedule: `colorfulness_mean` was lowest at the earliest
checkpoints (30.9 @ step 20, climbing to 47.4 @ step 100) under `lr_scheduler: constant` with **no
warmup** — this rank-8 LoRA, trained on only 21 triples, takes the full `5e-4` step from iteration 1.
That's a plausible mechanism for early-training softness that eases as training continues, distinct
from the harder exp062-run-2 collapse (near-blank frames, not just soft ones).

## Setup
Added `lr_warmup_steps: int = 0` to `unlearn_frame_replace.py`'s `Config` — a linear ramp from ~0 to
`learning_rate` over N steps (`torch.optim.lr_scheduler.LinearLR`, chained via `SequentialLR` in
front of whatever `lr_scheduler` is set to; `LinearLR` alone when `lr_scheduler: constant`, since it
holds at the end factor — full LR — once `total_iters` is exceeded). `0` reproduces the exact
pre-existing behavior (no warmup), so this is purely additive — no change to exp073 or earlier runs.

Grid over `lr_warmup_steps: [0, 30, 60]`, otherwise identical to exp073 (same 21-triple dataset,
regime, `steps: 100`, `save_interval: 20`, eval sets, seed). `0` is included as an in-run control
so the comparison to 30/60 isn't confounded by run-to-run GPU/queue noise (exp064's analysis found
classification/generation isn't perfectly bit-reproducible across GPU allocations) — even though
exp073 already covers the `lr_warmup_steps=0` point.

## What to watch
- `colorfulness_mean` / `clip_score_mean` at matched steps across the three warmup values — does a
  longer warmup raise early-checkpoint colorfulness the way exp073's own step-20-to-100 climb
  suggests it might?
- `concept_detection_rate` at matched steps — warmup should slow the erasure descent somewhat (the
  effective LR is lower early on), so don't expect step-20 detection to still be 0.0 the way
  exp073's un-warmed-up run showed; the real test is whether erasure still lands by step 100 while
  looking sharper.
- Visual review of the actual clips (the only way to judge "sharper," per exp073's lesson that our
  metrics don't measure this) — same caveat as exp073/exp074: don't declare success from metrics
  alone.

## Status
- [ ] Submitted.
- [ ] Aggregate metrics compared across the three `lr_warmup_steps` values (matched steps).
- [ ] Visually reviewed — does warmup actually look sharper, and does erasure still land?
- [ ] Decide: adopt warmup as the new default regime for nudity runs, or rule it out.
