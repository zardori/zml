---
status: ready
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  Not yet run.
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

## Status
- [ ] Submitted.
- [ ] Live monitor checked: top-1 reaches 0.00 and holds, top-5 trajectory noted against the
      0.11-0.28 band every rank-8 run has hit.
- [ ] If healthy, full `esr_psr` eval queued as the next experiment number. If not, report as
      inconclusive/abandoned and flag the rank lever as needing a different approach than either
      exp141's or this run's recipe.
