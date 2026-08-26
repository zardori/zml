---
status: done
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  CONVERGES CLEANLY, LIVE MONITOR SUPPORTS BOTH FOLLOW-UP EVALS. Concept top-1 is 0.00 at every
  checkpoint from step 100 through 600 (one blip of 0.01 at step 500) — as fast as exp142
  (rank 32, random-distractor) and exp157 (rank 8, void). Top-5 is noisy and does NOT settle
  monotonically: 0.02 (step 100) -> 0.00 (200) -> 0.00 (300) -> 0.13 (400) -> 0.10 (500) -> 0.12
  (600) — per this thread's standing lesson (exp135, exp139, exp142) a clean live top-5 read is
  not a reliable predictor either way, so this is a lead, not a result. Concept motion does NOT
  collapse the way it did for rank 32 on the random-distractor dataset (exp142's live read:
  0.240 -> 0.061, later confirmed by exp143's full protocol at 0.223): here it oscillates
  0.712/0.085/0.202/0.285/0.076/0.396 and the FINAL checkpoint (0.396) is well clear of the 0.15
  guard floor, closer to exp157/exp158's healthy void-dataset motion story (rank 8: 0.690) than to
  exp142/exp143's rank-32/random-distractor pattern — though step 200 and step 500 dip to
  0.085/0.076, both already below the 0.15 floor in this live sample, a reminder that the full
  200-prompt protocol is the only number that settles it. Two full esr_psr evals queued:
  exp161 (step 600, the "no deviation" default, testing whether void+rank32 beat exp143's ESR-5
  20.92 AND exp158's ESR-1/PSR 70.92/85.95/95.44 at once) and exp162 (step 300, testing whether
  exp150's rank-32 early-stop optimum, 38.67 ESR-5, reproduces on the void dataset).
---
# exp160 — frame_replace chain-saw erasure: void-target dataset x rank 32, CogVideoX-2B

## Why
Two levers have now each moved a different half of the GOAL.md target, independently and cleanly:

- **Void-target prompt_b** (exp157 train / exp158 eval, rank 8, exp156's dataset): restricted
  ESR-1 49.90 → 70.92 (+21.02), PSR-1 82.71 → 85.95 (+3.24), PSR-5 93.19 → 95.44 (+2.25), and the
  best erased-class motion margin of any rank-8 checkpoint (0.690 vs base 0.840, -18%). But ESR-5
  is flat (15.61 → 15.41) and chain saw's own restricted top-5 barely moves (0.844 → 0.846) — the
  residual-signal problem (the object staying in the model's top-5 guess even after top-1 erasure)
  is untouched.
- **Rank 32 + early stop** (exp142/exp143 train/eval, then exp150's step-300 read, exp131's
  dataset): restricted ESR-5 15.61 → 20.92 (rank alone, step 600) → 38.67 (step 300 early stop) —
  this thread's best ESR-5 by a wide margin, with PSR holding or improving alongside it.

These two experiments changed different, non-overlapping things (prompt_b's content vs. LoRA
capacity) and moved different cells of the same eval. Nothing has tested them together.

## Hypothesis and what would falsify it
Hypothesis: training exp156's void-target dataset at rank 32 (recipe otherwise identical to
exp142's rank-32/exp131 run — lr 0.0005 unscaled, 600 steps, save_interval 100) converges as
cleanly as exp157 (rank 8, same dataset) and exp142 (rank 32, exp131's dataset), and a full
esr_psr eval on the resulting checkpoint(s) beats exp158's ESR-1/PSR row AND approaches exp150's
ESR-5 (38.67) — i.e. the two levers' gains add rather than one cannibalizing the other.

Falsified by (checked from the live 9-prompt monitor, before spending a full eval):
- Non-convergence — concept top-1 not reaching 0.00 by step ~100-200, the exp141 failure mode.
  Ruled out as a confound here because lr/steps are unchanged from exp142's already-converging
  recipe; a failure would mean rank x void-target interact badly, not a repeat of exp141's
  lr-scaling bug.
- If it does converge, the earlier failure this thread has repeatedly hit (exp135, exp139,
  exp142's OWN live top-5 read) is a clean live monitor that does NOT translate to a full-protocol
  ESR-5 gain — so per standing practice, only a live-monitor read that is at least as healthy as
  exp142's own (which DID confirm) queues a full eval; a merely-adequate read does not.

## Setup
Field-for-field exp142 (rank 32, lr 0.0005, 600 steps, save_interval 100, exp132's 2B preservation
anchors, eta 2.0) except `metadata_file`/`latents_dir`, which point at exp156/exp157's void-target
dataset (single source, no exp139 extra_sources merge — matching exp157's dataset exactly, not
exp142's merged 47-row set) instead of exp131's random-distractor dataset.

## What to watch
- Live top-1/top-5/motion convergence, same read as every prior training run in this thread.
- If healthy: queue a full esr_psr eval next tick on the final checkpoint (or, per exp149-155's
  precedent, an earlier checkpoint if the live monitor names one) against exp158 (void/rank8) and
  exp150 (random-distractor/rank32/step300) cell for cell.

## Status
- [x] Submitted.
- [x] Live monitor checked for healthy convergence — clean, see takeaway.
- [x] Full esr_psr eval queued: exp161 (step 600) and exp162 (step 300, per exp149-155 precedent).
