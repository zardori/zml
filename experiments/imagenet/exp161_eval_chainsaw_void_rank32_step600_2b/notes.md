---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  HYPOTHESIS FALSIFIED ON ESR, PARTIALLY CONFIRMED ON PSR: THE TWO LEVERS DO NOT ADD, THEY
  INTERFERE. Restricted (10-way) row: ESR-1 45.71, ESR-5 10.82, PSR-1 90.25, PSR-5 97.07. Against
  exp143 (rank 32, random-distractor, step 600: 67.86 / 20.92 / 85.28 / 93.92) ESR-1 is DOWN 22.15
  and ESR-5 is DOWN 10.10 — both well below exp143's read, not at or above it as hypothesized.
  Against exp158 (rank 8, void, step 600: 70.92 / 15.41 / 85.95 / 95.44) ESR-1 is DOWN 25.21 and
  ESR-5 is DOWN 4.59 — also below, not at or above. So on erasure this checkpoint is worse than
  EITHER single lever alone, not the sum of both — combining void-target data with rank-32
  capacity actively hurts erasure rather than stacking the two gains. PSR does partially confirm:
  PSR-1 90.25 and PSR-5 97.07 both clear exp158's PSR-1/PSR-5 (85.95/95.44), the best PSR-1/PSR-5
  pairing in the whole thread — but that is the mirror image of weak erasure (less concept removed
  → less collateral damage → higher preservation), not a genuine "preservation win" independent of
  the ESR loss. Chain saw's own restricted top-5 is presumably still high given ESR-5 10.82 (near
  base). Net: the two levers that each independently moved a different axis of GOAL.md's target
  cannibalize each other when stacked at rank 32/step 600. exp162 checks whether an earlier
  stopping point (step 300, mirroring exp150's rank-32 early-stop optimum) rescues this, and
  whether the decline continues toward even earlier checkpoints (exp163/exp164, step 200/step 100,
  queued this tick).
---
# exp161 — eval: chain-saw void-target dataset x rank 32, CogVideoX-2B, final checkpoint (step 600)

## Why
exp160 trained exp156's void-target dataset at rank 32 (recipe otherwise identical to exp142) to
test whether two independently-established levers add: void-target's ESR-1/PSR gain (exp157/158,
rank 8) and rank 32's ESR-5 gain (exp142/143). Live monitor converged cleanly — top-1 0.00 from
step 100, top-5 noisy but low (0.02/0.00/0.00/0.13/0.10/0.12 across steps 100-600), and concept
motion at step 600 is 0.396, a healthy margin above the 0.15 guard floor and well clear of the
collapse rank 32's own random-distractor live read showed (exp142: 0.240 → 0.061).

## Hypothesis and what would falsify it
Hypothesis: this checkpoint's full esr_psr row is at or above exp158's ESR-1/PSR-1/PSR-5
(70.92 / 85.95 / 95.44) AND at or above exp143's ESR-5 (20.92) — i.e. the two levers' gains stack
rather than one cannibalizing the other.

Falsified by: any cell falling below the WORSE of exp143's and exp158's corresponding reading
(worse in the direction that matters — lower ESR, lower PSR). Partial confirmation (some cells add,
others don't) is also a real, reportable outcome, not a failure of the run.

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp160's `frame_replace_lora_step600` checkpoint,
identical 200-prompt protocol to every other row in this thread.

## Status
- [ ] Submitted.
- [ ] Compared against exp143 (rank 32, random-distractor, step 600) and exp158 (rank 8, void,
      step 600) cell-for-cell.
