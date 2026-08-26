---
status: ready
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  Not yet run.
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
