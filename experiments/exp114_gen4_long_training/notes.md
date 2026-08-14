---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  exp110 for 500 steps instead of 200, everything else identical. Motivated by exp110's loss still
  falling at the cut-off (0.489 -> 0.400, loss_erase -21%, loss_retain flat). Prior is that it ends
  WORSE: exp062's three long arms all decayed monotonically past step 100, and exp110 already
  rebounds from 0.0000 at step 140 to 0.12 by 200. Worth one job because exp110's U is far flatter
  than anything before it and nobody has looked past 200 steps on good data. 1 job.
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
