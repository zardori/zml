---
status: done
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  PRIOR CONFIRMED: longer training does not help. Best points are steps 100-140 (rate 0.000-0.010),
  exactly where exp110 peaked; past 200 the rate climbs to 0.19-0.27 while clip score decays
  0.29 -> 0.25 and motion falls to 0.02-0.17. Also a clean REPRODUCIBILITY check — the first 200
  steps track exp110 within 0.01-0.04 at every shared checkpoint, so the trajectory is stable and
  the step-count question is closed with evidence.
---
# exp114 — gen4, 500 steps

## Why
[exp110](../exp110_frame_replace_nudity_gen4/notes.md)'s training loss had not converged when the run
stopped. Its four logged windows (`metrics_log_interval: 50`):

| step | 49 | 99 | 149 | 199 |
|---|---|---|---|---|
| `train/loss` | 0.4891 | 0.4454 | 0.4573 | **0.4000** |
| `train/loss_erase` | 0.4075 | 0.3594 | 0.3761 | **0.3207** |
| `train/loss_retain` | 0.0816 | 0.0860 | 0.0813 | 0.0793 |

Down 18% overall with the minimum at the last window, `loss_erase` down 21%, and retention flat —
the model is still learning the erase objective at the point it is cut off. Four points is thin and
step 149 bumps upward, so "steadily" overstates it, but "not converged" is well supported.

## The prior is that this makes things worse

Train loss and eval erasure have been **anti-correlated** in this method. exp062 is the only
long-run evidence in the thread, and all three arms decayed monotonically:

| step | 100 | 200 | 300 | 400 | 600 |
|---|---|---|---|---|---|
| arm 1 | 0.07 | 0.17 | 0.58 | 0.53 | 0.64 |
| arm 2 | 0.33 | 0.62 | 0.49 | 0.61 | 0.69 |
| arm 3 | 0.31 | 0.26 | 0.50 | 0.31 | 0.37 |

exp110 shows the same shape in miniature: 0.0000 at step 140, 0.21 at 150, 0.23 at 170, settling at
0.12. So the honest expectation is that step 500 is worse than step 140.

**Recording that here before the run** so a negative result reads as a confirmed prior rather than a
surprise — and so a *positive* one is properly surprising.

## Why run it anyway
Two things make it a real question rather than a foregone conclusion:

1. **exp110's U is far flatter than any previous run.** Its rebound peaks at 0.23 where every other
   arm reached 0.49-0.76, and its late window (160-200) holds rate 0.124 at *full* colour recovery.
   A flatter U might keep flattening, or might have a second descent — nobody knows.
2. **exp062 is two dataset generations old.** It ran on gen1 data, before the frozen-donor fix
   (exp087) and before the wardrobe rewrite (exp109) that produced the only real improvement this
   thread has seen. Its long-run behaviour is not obviously exp110's.

It is one job, and either outcome closes the step-count question for the paper with evidence rather
than an assumption.

## Design
Identical to exp110 in every field except `steps` (200 -> 500) and `save_interval` (10 -> 20). Same
data, same fire retention, same eta 2.0, same lr 1e-4, same `global_seed: 42`.

**Trained fresh, not resumed** — the unlearn entrypoint has no resume path, and a fresh run means the
first 200 steps should reproduce exp110's trajectory. Where they diverge is a free reproducibility
check on a result the paper now depends on.

`save_interval: 20` costs resolution inside exp110's good window (90-140 is sampled at 100/120/140)
but this run is asking about steps 200-500, and 25 evals keeps the job near exp110's ~8.5h for 200
steps + 20 evals.

## What to watch
- **Whether anything past step 200 beats step 140's 0.0000 / colour 35.4 / motion 0.25.** That is the
  bar. A lower rate at similar quality would be a genuine find; anything else confirms the prior.
- **The 20-step-aligned overlay against exp110** over the shared 0-200 range. Divergence there means
  the run is not reproducible and both results need re-examining before either is reported.
- **Two adjacent checkpoints, never one.** Isolated zeros have been misread as regimes three times in
  this thread.
- **Late-window aggregate (steps 400-500)** rather than any single n=10 read.

## Status
- [ ] Submitted (1 job).
- [ ] Overlay against exp110 over steps 0-200 (reproducibility).
- [ ] Anything past 200 compared against exp110 step 140.
- [ ] DOVER scored locally if a checkpoint here is a candidate.


## Results (2026-08-14) — the prior held

| step | 60 | 100 | 120 | 140 | 200 | 280 | 340 | 420 | 480 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|
| rate | 0.09 | **0.00** | **0.00** | 0.01 | 0.16 | 0.01 | 0.10 | 0.25 | 0.19 | 0.27 |
| colour | 22.2 | 28.5 | 31.2 | 36.3 | 36.3 | 29.9 | 31.6 | 28.7 | 22.2 | 30.3 |
| motion | 0.19 | 0.22 | 0.25 | 0.28 | 0.16 | 0.16 | 0.15 | 0.15 | 0.02 | 0.17 |
| clip | 0.30 | 0.27 | 0.29 | 0.29 | 0.28 | 0.26 | 0.25 | 0.25 | 0.25 | 0.26 |

**Nothing past step 200 beats step 140.** The best window is steps 100-140, exactly where exp110
peaked, and the run then drifts up to 0.19-0.27 while clip score decays monotonically 0.29 -> 0.25
and motion collapses to 0.02 by step 480. The isolated 0.01 at step 280 is a single checkpoint
between 0.10 and 0.10 — a transient, not a regime.

This confirms exp062's pattern on two-generations-newer data and closes the step-count question:
**train loss falling is not evidence that more steps help.** exp110's loss was still descending at
step 200 and the extra 300 steps bought nothing.

### Free reproducibility check

Over the shared range, against exp110 (fresh run, same seed, same data):

| step | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|
| exp110 | 0.01 | 0.00 | 0.00 | 0.06 | 0.10 | 0.12 |
| exp114 | 0.00 | 0.00 | 0.01 | 0.09 | 0.11 | 0.16 |

Within 0.01-0.04 everywhere. The trajectory is stable across runs, which matters because it means
exp112's contradiction of exp110 is **not** run-to-run noise — it is purely the n=10 vs n=100
sampling difference.

## Status
- [x] Submitted and complete (1 job, 500 steps).
- [x] Overlay against exp110 over steps 0-200 — reproducible within 0.01-0.04.
- [x] Nothing past 200 beats step 140. Step-count question closed.
- ~~DOVER~~ — no checkpoint here is a candidate, so not scored.
