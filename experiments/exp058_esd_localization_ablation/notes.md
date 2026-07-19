# exp058 — ESD+preservation with vs. without mask-based localization (fire)

## Goal
Direct, controlled comparison of the T2VUnlearning localization regularizer `L_loc`
(arXiv:2505.17550, Eq. 9) against the plain `esd_preservation` recipe. exp054 produced
with-localization numbers but there was no no-localization run under the *current* eval code to
compare against (exp008 predates `metrics.json`), so this reruns both arms together.

## Setup
Two-run grid over `use_localization` only; every other hyperparameter and `global_seed: 42` are
held fixed, so the arms differ solely by the `L_loc` term.

- `run_001`: `use_localization=false` — baseline (exp008 recipe: rank 8, lr 5e-4, ng 1.0, 1000 steps).
- `run_002`: `use_localization=true`  — baseline + `L_loc`, weight 1.0, mask word "fire", soft mask.

## What to watch
- `train/loss_loc` (run_002 only) should trend down; `loss_forget`/`loss_preserve` should track
  run_001 reasonably closely if localization is doing something targeted rather than global.
- Control-set `metrics.json` at each `eval_step_*`, both arms: `concept` fire scores should drop,
  while `unrelated` (and `related`) stay near base. The claim to test is that run_002 gets a
  *better erasure/quality trade-off* — i.e. comparable concept erasure with less collateral damage
  to the non-concept sets — not merely more erasure.

## Results
_TBD_
