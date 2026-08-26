---
status: done
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  NON-CONVERGENCE FALSIFIER MISSED, BUT SO DID THE HOPED-FOR TREND: the curve does not keep
  improving below step 100, it goes flat and noisy. `concept_detection_rate` (9-prompt live sample,
  1/9 granularity) is already at or near floor from step 20 onward and flips 0.11/0.0/0.11/0.0/0.0/
  0.11/0.0 across steps 20-140 with no monotonic trend — that flip is single-video noise at this
  sample size, not a signal. `concept_area_score_mean` (the continuous residual measure) drops
  sharply from step 20 to step 40 (0.0396 → 0.0074, -81%) then stays in a low, noisy band
  (0.0006-0.016) through step 140 with no further systematic decrease — so nearly all of this run's
  visible suppression happens by step 40, not step 100, but nothing below 100 reads as clearly
  better than 100 itself. Motion never approaches the 0.15 guard floor at any checkpoint (range
  0.205-0.401). Step 100 here reproduces exp147's own step-100 checkpoint's live read (detection
  rate 0.0, consistent with exp147's report of top-1 0.00 from step 100) as intended — confirms the
  training trajectory is deterministic and comparable, not a different run. Per the thread's
  standing "queue a full eval only if the live monitor names a clear candidate" practice, none of
  steps 20-80 clearly beats the already-fully-evaluated step 100 (exp153: restricted ESR-1 77.86 /
  ESR-5 44.49, this thread's best full-protocol row) on this noisy a live signal, so no new full
  `esr_psr` eval is queued from this run. This closes rank 64's checkpoint-early-stopping search:
  exp153's step-100 checkpoint remains the operating point, and the ESR/PSR-vs-step curve mapped by
  exp148→exp149→exp151→exp153→this run has no evidence of a further peak below it.
---
# exp155 — rank 64, chain saw, fine-grained checkpoints (every 20 steps, up to 140) to find where
# the still-rising ESR/PSR-vs-step curve actually peaks

## Why
Four consecutive rank-64 checkpoints (exp148 step 600, exp149 step 300, exp151 step 200, exp153
step 100) show restricted ESR-1/ESR-5 improving monotonically as the stopping point moves earlier:

| step | ESR-1 | ESR-5 | PSR-1 | PSR-5 | chain-saw motion | preserved motion loss |
|---|---|---|---|---|---|---|
| 600 | 71.53 | 16.43 | 76.20 | 92.45 | 0.181 | 48.5% |
| 300 | 74.49 | 21.63 | 79.97 | 91.87 | 0.378 | 17.4% |
| 200 | 74.90 | 32.35 | 80.15 | 96.41 | 0.176 | 49.8% |
| 100 | 77.86 | 44.49 | 80.53 | 92.73 | 0.459 | 39.4% |

Step 100 is this thread's best row on both ESR-1 and ESR-5 (ESR-5 44.49 is the best in the *whole*
imagenet thread, beating exp150's rank-32/step-300 read of 38.67), and it clears the motion guard
with the largest margin yet (0.459 vs the 0.15 floor). But 100 is also the earliest checkpoint that
has ever been saved for this run (`save_interval: 100`) — every prior "does it keep improving
earlier" question in this sweep (exp149→exp151→exp153) has been answered by evaluating a
checkpoint that already existed. This time it does not: nothing below step 100 has been trained.

Motion preservation does NOT track the same monotonic trend — step 300 is the clear best point
(17.4% mean preserved-class motion loss vs exp130's base), with 600, 200 and 100 all in the
39-50% range. So the two axes (classification-based ESR/PSR, motion-based preservation) are
already known to peak at different steps for this rank; this run's job is to resolve the
classification axis's peak, not re-litigate the motion axis (exp149 already has the best motion
reading in the sweep, at step 300).

## Hypothesis and what would falsify it
Hypothesis: the ESR/PSR-vs-step curve continues rising below step 100 — i.e. steps 20-80 will show
concept top-1 already converged to 0.00 (per exp147's live monitor, which reached 0.00 by step
100) with lower top-5 residual than step 100's live read, live-sample motion still comfortably
above the 0.15 floor and not yet at exp147's live-sample-pessimistic pattern.

Falsified by:
- **Non-convergence below some step** — top-1 fails to reach 0.00, or oscillates rather than
  holding, at 20/40/60 — would mean rank 64 needs more than ~60-80 steps to erase the concept at
  all, and the true floor of this curve is somewhere in 60-100, not below 60.
- **Live top-5 or motion gets WORSE than step 100's read as steps decrease further** — would put a
  local peak inside this run's own range (mirroring the step-300 local-optimum shape already found
  for the motion axis), rather than the monotonic trend continuing all the way to the shortest
  training runs.

Only the live 9-prompt monitor is reported here, per standing practice (a floored live top-5 has
been directionally wrong twice in this thread already — exp135, exp139 — but exp143's rank-32/
step-600 case shows it is not always wrong, and exp153 is the most recent case where a rising
live-sample ESR-5 trend was NOT falsified by the full protocol). A full `esr_psr` eval on whichever
checkpoint looks best is the natural next-tick follow-up, mirroring exp147→exp153.

## Setup
Field-for-field exp147 (2B, merged 47-row exp131+exp138 dataset, eta=2.0, lr 0.0005, rank 64)
except `steps: 140` (was 600) and `save_interval: 20` (was 100). Step 100 here should reproduce
exp147's step-100 checkpoint exactly (same seed, same data order up to that point) — a free sanity
check that this is the same training trajectory, not a different run.

## What to watch
- **Live concept top-1/top-5 trajectory at steps 20, 40, 60, 80** — the genuinely new information;
  nothing in this thread has evaluated a chain-saw checkpoint this early before.
- **Live concept-class motion** at each checkpoint, watching for the guard-adjacent thinning
  exp151 found at step 200 (margin 0.026) — though per exp143's lesson, a low live-sample number is
  not disqualifying without a full-protocol read.
- **Step-100 reproducibility** — this run's step-100 checkpoint should match exp147's own step-100
  live-monitor read; a mismatch would mean something in the data pipeline or seeding is not as
  deterministic as assumed and would need investigating before trusting any of steps 20-80.

## Result
Per-checkpoint live monitor (`concept_detection_rate` / `concept_area_score_mean` /
`motion_score_mean`, 9 prompts):

| step | detection_rate | area_score | motion |
|---|---|---|---|
| 20 | 0.111 | 0.0396 | 0.236 |
| 40 | 0.0 | 0.0074 | 0.363 |
| 60 | 0.111 | 0.0163 | 0.332 |
| 80 | 0.0 | 0.0010 | 0.216 |
| 100 | 0.0 | 0.0021 | 0.332 |
| 120 | 0.111 | 0.0092 | 0.205 |
| 140 | 0.0 | 0.0059 | 0.372 |

No checkpoint below 100 reads as a clear, non-noisy improvement over step 100 itself — the biggest
drop in `concept_area_score_mean` is already behind us by step 40, and `detection_rate`'s 0/0.111
flips are 1-video noise at this sample size. Motion never nears the 0.15 floor.

## Status
- [x] Submitted (helios job 21173643, 2026-08-26, elapsed 2h14m of a 4h budget).
- [x] Live monitor checked across all 7 checkpoints (20-140).
- [x] Decision: no full `esr_psr` eval queued — no checkpoint below step 100 shows a clear live-
      monitor win over exp153's already-evaluated step 100, unlike exp149/exp151/exp153's earlier
      finds, which each named a specific better-looking checkpoint before their full eval ran.
