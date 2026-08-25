---
status: done
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  CONVERGES, BUT THE LIVE MONITOR CAN'T TELL US MORE THAN EXP142 ALREADY DID. Live concept top-1
  is 0.00 from step 100 (even faster than rank 8/32's step ~100-200) and holds through step 600,
  with one transient blip at step 400 (top-1 0.11, top-5 0.33) that resolves by step 500. Final
  top-5 is 0.00, matching exp142's already-near-zero band (0.00 at 4 of 6 checkpoints) rather than
  dropping further — by the notes' own pre-registered falsifier this reads as "no further
  live-sample gain over exp142," except exp139 (rank 8) also bottomed its live top-5 at 0.00 at
  two checkpoints and exp142 (rank 32) still won +5.31 ESR-5 on the full protocol over it. So a
  9-prompt sample floors at 0 well before the real ESR-5 differences between ranks stop moving —
  the live monitor cannot discriminate rank 32 from rank 64 either way, only rule out
  non-convergence. Live concept motion oscillates hard (0.127 / 0.017 / 0.237 / 0.098 / 0.243 /
  0.089) and ends at 0.089, under GOAL.md's 0.15 guard floor — same pattern as exp142's live read
  (0.061 final) that exp143's full eval then cleared at 0.223, so per that precedent this is not
  read as disqualifying without the full-protocol number. Runtime 3.86h (13901s) on helios, same
  ballpark as exp142's 3.7h — confirms rank 64 adds no measurable per-step overhead either, same as
  the rank 8->32 step. Convergence is healthy, so exp148 spends the full esr_psr eval per this
  thread's standing "queue a full eval only if the live monitor is healthy" gate (exp142->exp143).
---
# exp147 — frame_replace erasure of CHAIN SAW on CogVideoX-2B, LoRA rank 64 at exp139/exp142's UNSCALED lr/steps

## Why
exp143 (the full-protocol eval of exp142's rank-32 checkpoint) found capacity is a real, non-null
lever — the first one in this thread's three-lever search (eta: exp137, closed null / traded
preservation; dataset size: exp140, closed null; capacity: exp142/exp143). Restricted ESR-1 moved
49.90 -> 67.86 (+17.96), ESR-5 15.61 -> 20.92 (+5.31), and *both* PSR cells also improved
(85.28 vs 82.71, 93.92 vs 93.19) — no trade-off, unlike eta. Chain saw's own restricted top-5
dropped to 0.79, the first read below the ~0.84-0.85 floor every rank-8 arm was stuck at.

But the run still lands well short of GOAL.md's target (ESR-1 92.38, ESR-5 77.09 — a 56.17-point
gap on ESR-5, the binding guard). This run tests whether a second doubling (32 -> 64) continues to
close that gap at a similar rate, plateaus, or reverses — at the same lr/step budget that has now
converged cleanly at rank 8 (exp139) and rank 32 (exp142), removing any lr-scaling confound the way
exp142 removed it for exp141's failed rank-32 attempt.

## Hypothesis and what would falsify it
Hypothesis: at exp139/exp142's exact lr (0.0005) and step budget (600), rank 64 converges concept
top-1 to 0.00 at least as fast as rank 8/32 did (by step ~100-200), and the ESR-5 gain continues in
the same direction — either by top-5 dropping further on the live sample, or (more decisively) on a
follow-up full eval, matching exp142's pattern of "queue a full eval only if the live monitor is
healthy."

Falsified by:
- **Non-convergence** — top-1 fails to reach 0.00 by step 600 or oscillates the way exp141's
  mis-scaled rank-32 attempt did. Would mean rank 64 needs a different lr/step budget before it can
  be read at all, the same open question exp141 raised and exp142 closed for rank 32.
- **Convergence, but no further live-sample gain over exp142's read** (top-5 not lower than
  exp142's steps 200-600 near-zero band, motion not meaningfully different) — would suggest the
  capacity lever is already saturating around rank 32, and doubling again is not where the
  remaining ESR-5 gap gets closed.

Only the live 9-prompt monitor is reported here, per standing practice (exp071/exp133/exp139's
lesson that small live samples are directional, not the protocol — though exp143 is this thread's
first case where a live-sample "top-5 hits 0.00" signal was NOT falsified by the full eval, so this
run's live read carries more weight than exp135's or exp139's did going in). A full `esr_psr` eval
is the natural next-tick follow-up if the live top-1 trajectory is healthy.

## Setup
Field-for-field exp142 (2B, merged 47-row exp131+exp138 dataset, eta=2.0, lr 0.0005, steps 600,
save_interval 100) except `lora_rank: 64`, `lora_alpha: 64.0` (was 32/32.0) — capacity, the sole
variable under test, isolated exactly as exp142 isolated it from exp141's lr-scaling confound.

## What to watch
- **Live concept top-1/top-5 trajectory** — convergence speed and whether top-5 drops further than
  exp142's near-zero read (steps 200, 300, 500, 600 all 0.00 there).
- **Live concept-class motion** — exp142's live sample fell to 0.061 (under the 0.15 guard floor)
  but the full-protocol number (exp143) came in at 0.223, comfortably above floor. Watch whether
  rank 64 repeats that same live-sample-pessimistic pattern, and don't read a low live number as
  disqualifying without the full eval, per exp143's own lesson.
- **Runtime** — `slurm_time` budgeted with extra headroom (8h vs exp142's 6h) on the chance rank 64
  is where per-step overhead stops being negligible; worth confirming either way.

## Status
- [x] Submitted. Completed on helios, job 21082439, 13901s (3.86h), exit 0.
- [x] Live monitor checked: top-1 converges even faster than rank 8/32 (0.00 from step 100, one
      blip at step 400 that resolves by 500) and top-5 ends at 0.00, matching — not beating —
      exp142's already-near-zero band. The falsifier's second clause technically fires ("no further
      live-sample gain over exp142"), but exp139->exp142 already showed a floored live top-5 can
      still hide a real full-protocol ESR-5 gain, so this is read as inconclusive-at-this-resolution,
      not as evidence the lever has saturated.
- [x] Decision: convergence is healthy, so per the exp142->exp143 gate a full `esr_psr` eval is
      queued as exp148.
