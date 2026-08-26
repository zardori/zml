---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
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
