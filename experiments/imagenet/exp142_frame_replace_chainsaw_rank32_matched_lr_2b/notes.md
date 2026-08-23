---
status: done
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  CONVERGES CLEANLY, UNLIKE exp141 — the confound was the lr scaling, not capacity. Live concept
  top-1 reaches 0.00 by step 200 and holds through step 600 (one transient 0.09 at step 100), same
  pace as every rank-8/eta-2.0 run (exp133, exp135, exp139). Top-5 — the metric every prior lever
  (eta: exp137, dataset size: exp140) failed to move — lands at 0.00 at four of six checkpoints
  (200, 300, 500, 600) with only tiny blips at 100 (0.11) and 400 (0.03), below the 0.11-0.28 band
  every rank-8 run's live sample has been stuck at. BUT this is the third time a live 9-prompt
  monitor has shown "top-5 hits 0.00" (after exp135 and exp139), and both prior instances were
  falsified on the full 200-prompt protocol (exp137, exp140) — reason for skepticism, not
  confidence, going into the eval. More concerning and NEW: concept motion collapses within this
  run's own live sample, 0.240 (step 100) -> 0.061 (step 600, -74%), already below GOAL.md's 0.15
  motion-guard floor and lower than any final-checkpoint reading from exp133/exp135/exp139's
  rank-8 runs (exp133's was 0.140 at step 600). Per this run's own pre-registered criterion (queue
  a full eval only if top-1 is healthy — it is), exp143 spends the last untested single-lever
  hypothesis (capacity) on the full protocol, flagged going in that it may fail the motion guard
  even if ESR-5 improves.
---
# exp142 — frame_replace erasure of CHAIN SAW on CogVideoX-2B, LoRA rank 32 at exp139's UNSCALED lr/steps

## Why
exp141 tested LoRA capacity (rank 8 -> 32) but confounded it with an lr-scaling rule borrowed from
the nudity thread (lr *= 8/32 = 0.25, steps doubled 600 -> 1200 to compensate). It never
converged: live concept top-1 oscillated 0.07/0.31/0.30/0.11/0.11/0.20 across all six checkpoints
and never touched 0.00, where every rank-8/eta-2.0 run so far (exp133, exp135, exp139) hits 0.00 by
step ~150-200 and holds flat. exp141's own pre-registered falsifier says this reads as the scaled
lr undertraining rank 32 relative to its step budget, not as "capacity doesn't help" — so no full
`esr_psr` eval was spent on that checkpoint.

This run removes the confound instead of guessing a second correction (the path nudity's exp136 had
to take after its own first rank-32 attempt undershot). Every field is exp139's, field-for-field,
except `lora_rank`/`lora_alpha`. If rank 32 has a real advantage at the SAME lr/step budget that
already converges cleanly at rank 8, it should show up directly; if it still doesn't converge, that
is a much stronger statement about capacity than exp141 could make, because there is no lr-scaling
confound left to blame it on.

## Hypothesis and what would falsify it
Hypothesis: at exp139's exact lr (0.0005) and step budget (600), rank 32 converges concept top-1 to
0.00 at least as fast as rank 8 does (by step ~100-200), and — being the actual point of the test —
top-5 on the live set drops below the ~0.11-0.28 floor every rank-8 run has been stuck at, without
preserved-class motion or colourfulness moving worse than exp139/exp140's rank-8 readings.

Falsified by either:
- **Non-convergence at exp139's own recipe** — top-1 fails to reach 0.00 by step 600, or oscillates
  the way exp141 did. Since exp141's lr-scaling confound is now absent, this would be strong
  evidence that capacity is not the lever, independent of eta (exp137) and dataset size (exp140) —
  worth a `needs_human` on whether frame_replace itself, not a hyperparameter, is capped for this
  concept's residual top-5 signal.
- **Convergence at rank 8's pace, but top-5 lands in the same 0.11-0.28 band exp133-exp140 already
  hit** — capacity converges fine but buys nothing on the metric that matters, closing this lever
  the same clean way exp137 (eta) and exp140 (dataset) closed theirs.

Only the live 9-prompt monitor is reported here, per exp071/exp139's standing lesson that a small
live sample is not the protocol. A full `esr_psr` eval is queued as a follow-up only if the live
top-1 trajectory is healthy (reaches and holds 0.00) — otherwise this joins exp141 as inconclusive
and the rank lever needs a different correction before it can be read at all.

## Setup
Field-for-field exp139 (2B, merged 47-row exp131+exp138 dataset, eta=2.0, lr 0.0005, steps 600,
save_interval 100) except `lora_rank: 32`, `lora_alpha: 32.0` (was 8/8.0) — capacity, the sole
variable under test, isolated from the lr/step confound exp141 introduced.

## Results (2026-08-23) — converges cleanly; live top-5 signal strong but same shape as two already-falsified reads; new motion concern

Completed on helios (job 20962189, 3.7h of a 6h budget — same runtime as exp139's rank-8 run at
the same lr/step budget, confirming the slurm_time comment's prediction that rank alone adds no
large per-step overhead).

Live 9-prompt monitor, by checkpoint (step: top-1 / top-5 / concept motion / unrelated motion):

| step | top-1 | top-5 | concept motion | unrelated motion |
|---|---|---|---|---|
| 100 | 0.09 | 0.11 | 0.240 | 0.426 |
| 200 | 0.00 | 0.00 | 0.161 | 0.470 |
| 300 | 0.00 | 0.00 | 0.170 | 0.373 |
| 400 | 0.00 | 0.03 | 0.133 | 0.392 |
| 500 | 0.00 | 0.00 | 0.061 | 0.314 |
| 600 | 0.00 | 0.00 | 0.061 | 0.425 |

**Convergence**: clean, matching rank 8's pace exactly (0.00 by step 200, holding). This directly
answers exp141's open question — the earlier rank-32 run's oscillation (0.07/0.31/0.30/0.11/0.11/
0.20) was the lr-scaling confound, not a property of higher capacity. Rank 32 trains as stably as
rank 8 when the lr/step budget is left alone.

**Top-5** — the metric every previous lever left untouched — reads better than any rank-8 run's
live sample: 0.00 at steps 200, 300, 500, 600, against the 0.11-0.28 floor exp133/exp135/exp139
were stuck at. Taken alone this would be the first positive top-5 signal in the thread. It is not
taken alone: exp135's live top-5 also hit 0.00 and was falsified by exp137's full-protocol ESR-5
(10.31, *below* the eta=2.0 baseline); exp139's live top-5 also hit 0.00 at two checkpoints and was
falsified by exp140's full-protocol ESR-5 (15.82, statistically flat against baseline). Two out of
two prior instances of this exact signal did not survive N=200. This is the third instance, and
that base rate is the reason to stay skeptical rather than to read this table as a win.

**Motion — new and specific to this run.** Concept-class motion does not just stay flatter than
`unrelated`, as every prior run showed; it keeps falling to 0.061 by steps 500-600, a level neither
exp133 (0.140 final) nor exp139's live sample approached, and already below GOAL.md's 0.15 guard
floor *on the live sample itself*. If this holds on the full 200-prompt protocol it would fail the
motion guard outright, independent of whatever ESR-5 does — a different, capacity-specific version
of exp069/exp071's original freeze finding, not the eta- or dataset-scale-driven kind exp134/exp137/
exp140 already characterized (those all cleared the 0.15 floor comfortably, 0.26-0.39 on the erased
class at the full-protocol level).

## Status
- [x] Submitted (helios job 20962189, completed 2026-08-23T02:37).
- [x] Live monitor checked: top-1 reaches 0.00 by step 200 and holds through 600 — healthy by the
      pre-registered criterion. Top-5 trajectory is below the 0.11-0.28 band every rank-8 run hit,
      but this is the same shape as two signals (exp135, exp139) already falsified on the full
      protocol, so it is read with matching skepticism, not confidence.
- [x] Full `esr_psr` eval queued: exp143, on this run's `frame_replace_lora_step600` checkpoint —
      the top-1 trajectory is healthy per the pre-registered gate. Flagged going in that the
      concept-motion collapse to 0.061 in this run's own live sample (below the 0.15 guard floor)
      is a real risk the checkpoint fails the motion guard even if ESR-5 improves.
