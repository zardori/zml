---
status: done
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  TRAINING CONVERGES, AND THE LIVE MONITOR'S SHAPE IS QUALITATIVELY DIFFERENT FROM EVERY PRIOR
  RANK-8 RUN. Concept top-1 is 0.00 at every single checkpoint including step 100 (exp133's
  identical-recipe/random-distractor run took until step 200) and top-5 stays low and noisy
  (0.00/0.02/0.10/0.00/0.00/0.02 across steps 100-600), well under exp133's reported 0.11-0.22
  band. Unlike every prior "top-5 hits ~0" live read in this thread (exp135, exp139, exp142,
  exp147), concept motion does NOT collapse alongside it: it oscillates 0.69/0.32/0.64/0.61/
  0.27/0.56 with no monotonic decline and ends at 0.562 (step 600) -- far above exp133's own
  final read of 0.140 and nowhere near the 0.15 guard floor. Preserved-class ("unrelated")
  motion in the live sample drops 0.573 -> 0.322 (-44%) from step 100 to step 600, the opposite
  direction of exp133's live read (which rose, then was corrected downward by the full protocol)
  -- so this run's live sample is not repeating that specific optimism failure either. Net: this
  is the healthiest-looking live monitor yet for a rank-8 run, on a genuinely different axis
  (fast suppression without the motion collapse that co-occurred with strong suppression in
  every prior instance) -- but the thread's standing lesson is that a 9-prompt read like this
  has gone both ways on the full 200-prompt protocol (confirmed for exp142/rank-32, nulled for
  exp135's eta and exp139's dataset-size levers), so per exp157's own pre-registered falsifier
  this is a lead, not a result. Final checkpoint (frame_replace_lora_step600) sent to exp158 for
  the full esr_psr comparison against exp134's row (restricted ESR-1 49.90 / ESR-5 15.61 /
  PSR-1 82.71 / PSR-5 93.19), same "no reason to deviate from the final checkpoint" logic used
  throughout this thread since top-1 never wavers.

  CORRECTED 2026-08-26 by exp158: the live monitor's TOP-5 optimism did not survive the full
  protocol (restricted ESR-5 15.41 vs exp134's 15.61 -- flat, joining exp135/exp139's null
  instances, not exp142's confirming one), but its MOTION optimism did -- erased-class motion
  landed at 0.690 (base 0.840, -18%), the healthiest margin of any rank-8 checkpoint measured
  (exp134 0.390, exp137 0.371, exp140 0.262). Separately, ESR-1 moved a lot (49.90 -> 70.92,
  +21.02) and both PSR cells improved (PSR-1 +3.24, PSR-5 +2.25) -- the void target is a real,
  useful lever, just not on the axis (ESR-5) this run's live read predicted.
---
# exp157 — frame_replace chain-saw erasure trained on exp156's void-target dataset, CogVideoX-2B

## Why
Every single-lever sweep on this thread's baseline recipe has plateaued on ESR-5 in particular:
eta (exp137, worse), dataset size via a second independently-seeded batch (exp138→exp140, null),
LoRA rank (exp143/exp147/exp148, non-monotonic, peaks ~rank 32), checkpoint/early-stopping
(exp149-exp155, best full-protocol row exp153's rank-64/step-100 at restricted ESR-1 77.86 /
ESR-5 44.49). All of these tuned how hard or how long to push the same training *data*; none has
touched what the data itself teaches the model to substitute the concept with.

HINTS.md records that in the nudity thread, making prompt_b's target less varied (one consistent
look instead of a different outfit per row) made the unlearning signal clearer. exp156 built the
object-thread analogue — the same 30 prompt_a/prompt_c/seed rows as exp131, but prompt_b replaced
with a consistent void ("empty and bare") instead of a different random distractor object per row —
and screened at 25/30 (83%), 0 not-split, 5 no-concept: statistically identical yield to exp131,
and 4 of the 5 no-concept failures are the *same seeds* (3204, 3211, 3218, 3224) as exp131's, with
the other two (3203, 3226) flipping only because their conc_max sat within 0.01 of the 0.10 cutoff
in both runs — consistent with cross-run GPU nondeterminism, not a systematic prompt_b effect. That
is exactly what `split_mode: trajectory` predicts: the concept-half trajectory is generated
independently of prompt_b and spliced in afterward, so prompt_b's content cannot change whether
prompt_a renders the concept. exp156 confirmed the *build* transfers; this run tests whether it
changes what the LoRA *learns*.

## Hypothesis and what would falsify it
Hypothesis: training on the void-target dataset moves ESR-5 (the metric every other single lever
has failed to move) relative to exp134's reported row for the identical rank-8/eta-2.0/600-step
recipe on exp131's random-distractor dataset (restricted ESR-1 49.90, ESR-5 15.61, PSR-1 82.71,
PSR-5 93.19), without giving back PSR below GOAL.md's floors (54.03 / 82.14).

Falsified by: ESR-1/ESR-5 landing within noise of exp134's row — would mean prompt_b's diversity is
not a lever at this rank/eta/step budget, even though the nudity thread's version of the same idea
worked. A live 9-prompt read of "top-5 hits 0.00" is NOT sufficient evidence either way — exp135,
exp139 and exp147/153's live monitors all showed this and needed the full 200-prompt protocol
(exp137, exp140, exp143/148/149/151/153) to confirm or reverse it; the same discipline applies here.

## Setup
Field-for-field exp133 (which trained the same recipe on exp131's dataset) except
`metadata_file`/`latents_dir`, which point at exp156's screened set
(`exp156_split_chainsaw_void_2b/outputs_20260826_071200_screened.json` /
`.../outputs_20260826_071200/latents`) instead of exp131's. Rank 8, eta 2.0, lr 0.0005 (unscaled),
600 steps, exp132's 2B preservation anchors — all unchanged, so any ESR/PSR difference from
exp134's row is attributable to the training data's prompt_b content alone.

## What to watch
- Live 9-prompt concept top-1/top-5 and motion, same as every prior training run in this thread —
  read as a convergence check only, not as an ESR-5 prediction (see falsification note above).
- Once trained: queue the full `esr_psr` eval next tick and compare against exp134's row cell for
  cell. If ESR-5 moves clearly and PSR holds, the next question is whether the effect stacks with
  the rank-32/step-300 recipe (exp150, this thread's best ESR-5 so far at 38.67) — a second,
  later experiment, not this one.

## Status
- [ ] Submitted.
- [ ] Live monitor checked for healthy convergence (top-1 reaching ~0.00, no divergence).
- [ ] Full esr_psr eval queued and compared against exp134's row.
