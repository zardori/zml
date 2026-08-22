---
status: done
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  INCONCLUSIVE ON RANK, BY ITS OWN PRE-REGISTERED CRITERION -- the lr-scaling recipe undertrained,
  exactly the second falsifier this notes.md wrote down in advance. Live 9-prompt concept top-1
  never reaches 0.00 at any of the six checkpoints (0.07 / 0.31 / 0.30 / 0.11 / 0.11 / 0.20 across
  steps 200-1200) and oscillates rather than converging, where every rank-8/eta-2.0 run so far
  (exp133, exp135, exp139) hit 0.00 by step ~150-200 and held. Top-5 tracks the same non-convergence
  (0.24 / 0.66 / 0.72 / 0.48 / 0.49 / 0.38) -- it gets WORSE than the untrained baseline before
  drifting back down, never approaching 0. Per the pre-registered falsifier, this means the 8/32=
  0.25 lr scaling (borrowed from the nudity thread's exp129 rule, "hold effective step size
  constant") undertrained rank 32 relative to its 2x step budget -- it does NOT mean capacity
  doesn't help, and per the same pre-registration no full esr_psr eval was queued on this
  checkpoint. exp142 reruns rank 32 at exp139's exact lr (0.0005, unscaled) and step budget (600),
  isolating rank as the sole variable against a recipe already known to converge cleanly at rank 8,
  rather than guessing at a second scaling correction the way nudity's exp136 did.
---
# exp141 — frame_replace erasure of CHAIN SAW on CogVideoX-2B, LoRA rank 8 -> 32

## Why
Two levers are now closed on the reported metric (restricted ESR-5). exp137: pushing
`erase_esd_eta` 2.0 -> 3.0 made ESR-5 *worse* (10.31 vs exp134's 15.61) while costing more
preservation. exp140: nearly doubling the training set (25 -> 47 rows, exp139) moved ESR-1 by
+2.65 points (49.90 -> 52.55) and ESR-5 by +0.21 (15.61 -> 15.82) — both inside noise, and PSR-1
actually dropped 1.37 points. Every rank-8/eta-2.0 run converges to the same ceiling: restricted
ESR-1 ~50-54, ESR-5 ~15-16, chain-saw top-5 barely moves off base (0.84 vs base 1.0 in exp140) —
far short of the 92.38/77.09 target. Two independent levers, two null results: the ceiling is not
about eta or about how much data the LoRA sees.

The nudity thread is mid-way through the lever this thread has never touched: LoRA capacity.
`experiments/nudity/exp129_gen4_rank32_lr_scaled` (rank 32, lr scaled 1e-4 -> 2.5e-5 to hold the
effective step size) found its eta=2.0 arm still descending at step 200, past where rank-8 bottoms
out, at the same suppression rate but +9 colourfulness and no rebound — the "capacity substitutes
for push" signal — and its eta=4.0 arm undershot its decision point, forcing exp136 to rerun at
400 steps. That is read-only to this agent (nudity thread), but the recipe is a direct, cheap test
to import: hold eta and dataset fixed, raise rank, see if the LoRA has enough capacity to represent
a fuller edit than rank 8 can, closing the residual-top-5 gap without the preservation cost eta=3.0
paid.

## Hypothesis and what would falsify it
Hypothesis: raising `lora_rank` 8 -> 32 (alpha held at ratio 1, lr scaled by 8/32 = 0.25 to hold
the effective per-step update, steps doubled 600 -> 1200 so the smaller lr has room to converge),
holding eta=2.0, the merged 47-row dataset, and everything else at exp139's settings, moves
restricted ESR-5 clearly above exp140's 15.82 (and ideally ESR-1 well above 52.55) without PSR-1 /
PSR-5 / erased-class motion dropping below exp140's 81.34 / 93.30 / (this run's own quality block,
to be read against exp130's per-class base) or the 54.03 / 82.14 / 0.15 floors.

Falsified by:
- Restricted ESR-5 at or below exp140's 15.82 on the full 200-prompt eval (needs a follow-up eval
  once this training completes) — would mean the residual top-5 signal is not a capacity limit
  either, and the three obvious lever categories (erase pressure, data, capacity) are all
  exhausted at their current settings — worth a `needs_human` on whether the method itself, not a
  hyperparameter, is the ceiling for this concept.
- The live 9-prompt monitor failing to reach concept top-1 0.00 by step ~400 (double exp139's
  ~200, matching the 2x step budget) or oscillating instead of holding — would mean the scaled lr
  undertrained the run relative to its budget, not that rank doesn't help; would need a further
  step/lr correction before the rank hypothesis itself can be read, the same discovery nudity's
  exp136 made about its own first rank-32 attempt.

This run alone reports only the live 9-prompt monitor, per exp071/exp139's own lesson that a small
live sample is not the protocol. A full `esr_psr` eval is the follow-up this queues if the live
signal looks healthy — do not spend it if the live monitor is unhealthy per the second falsifier
above.

## Setup
Field-for-field exp139 (2B, merged 47-row exp131+exp138 dataset, eta=2.0) except:
- `lora_rank: 32`, `lora_alpha: 32.0` (was 8/8.0) — capacity, the variable under test.
- `learning_rate: 0.000125` (was 0.0005) — scaled by 8/32, same rule nudity's exp129 used ("holds
  effective step at rank-8 levels").
- `steps: 1200` (was 600), `save_interval: 200` (was 100) — widened up front so an undertrained
  rank-32 run cannot be mistaken for "capacity doesn't help"; nudity's exp136 discovered this
  correction was needed only after its first rank-32 run (exp129) undershot at unscaled steps.

Dataset, retention set, eta, timestep range, control prompts, eval cadence: all exp139's, so any
difference in the resulting live-monitor trajectory (and, downstream, ESR/PSR) is attributable to
rank/capacity, not a confound with the eta sweep (exp135/137) or the dataset-size test (exp138-140)
already run separately.

## What to watch
- **Concept top-1** on the live eval set — exp139 (rank 8, unscaled lr) reached 0.00 by step 200;
  with lr scaled to 1/4 and steps doubled, the equivalent point should land near step 400-800. A
  flat 0.00 well before step 1200 with room to spare is the health check.
- **Concept top-5** — the residual-signal metric every lever so far has failed to move
  (exp133-exp140 all settle 0.11-0.28 restricted top-5 or hold near base on chain saw itself). Any
  checkpoint that reaches and holds top-5 0.00 on the live set — the way exp139's did, before
  exp140 confirmed it on the full protocol — is the signal worth a full eval.
- **Colourfulness / DOVER-aesthetic on the concept clips** — the nudity thread's "capacity
  substitutes for push" reading was partly a colourfulness comparison (rank 32 held +9 over rank 8
  at the same suppression rate). Watch whether rank 32 avoids exp069/exp128's colourfulness spike
  (scene-destruction signal) at a given suppression level, not just whether it suppresses.
- **Preserved-class motion** on the 9-prompt unrelated sample — exp139's live sample dropped 31%
  (0.384 -> 0.266), closer to the true ~32% full-protocol loss exp134 measured than exp133's
  misleadingly optimistic +35% rise. Use exp139's read as the honest live-sample baseline, not
  exp133's.

## Status
- [x] Submitted (helios, job 20953347, completed 21674s / ~6.0h).
- [x] Live monitor checked: top-1 never reaches 0.00, oscillates 0.07-0.31 across all six
      checkpoints -- unhealthy per the pre-registered second falsifier.
- [x] Not healthy, so no full `esr_psr` eval queued on this checkpoint. See exp142, which reruns
      rank 32 at exp139's original, known-convergent lr/steps to isolate rank from the scaling
      recipe.
