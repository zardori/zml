---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
---
# exp134 — reported ESR/PSR for the chain-saw LoRA, on CogVideoX-2B

## Why
GOAL.md's target table is CogVideoX-2B, restricted (10-way) convention, compared against
T2VUnlearning's Table 4. exp133 trained the first 2B chain-saw `frame_replace` LoRA and its
9-prompt live eval reproduced exp069's 5b trajectory (concept top-1 0.09 → 0.00 by step 200,
holding through step 600). But exp071 already showed once, on 5b, that a small live-eval set can
mislead — exp069's monitor read the motion freeze as concept-conditional; the full 200-prompt
protocol showed it was global (nine preserved classes losing a mean 45% of their motion). exp133's
live eval shows the *opposite* surprise this time (unrelated-sample motion rising, not collapsing)
and that needs the same correction: only the full protocol tells us if it holds.

This run is the 2B counterpart of exp071 — same config shape, same checkpoint-selection logic,
new model_id and new `lora_checkpoint_dir` pointing at exp133 instead of exp069. It produces the
row that gets checked against GOAL.md's target table and every guard in it.

## Hypothesis and what would falsify it
Hypothesis: 2B's ESR/PSR row lands close to exp071's 5b row in shape — strong ESR-1/ESR-5 under
the 1000-way convention, a smaller but real gain under restricted, PSR held close to exp130's 2B
`Original` (restricted PSR-1 89.40, PSR-5 97.91) — and the GOAL.md motion guard (0.15 floor on the
erased class) either passes or fails the same way 5b's did (5b's own erased-class motion was 0.111,
already below the 0.15 floor exp071 helped calibrate).

Falsified by: restricted ESR-1/ESR-5 not clearing T2VUnlearning's bar (92.38/77.09) — expected on
a single checkpoint with no hyperparameter search, so not itself a failure of the method, just a
result to report honestly. More load-bearing: if the full-protocol preserved-class motion mean
matches 5b's collapse (~-45%) despite exp133's live sample suggesting otherwise, that confirms the
freeze is model-independent, not something 2B improved on; if it does NOT collapse, that is a
genuine 2B-specific finding worth its own note in `docs/imagenet_objects.md`.

## Setup
Field-for-field exp071 except:
- `model_id: THUDM/CogVideoX-2b`
- `lora_checkpoint_dir` points at exp133's final checkpoint
  (`experiments/imagenet/exp133_frame_replace_chainsaw_2b/outputs_20260820_181735/frame_replace_lora_step600`)
  instead of exp069's.
- `slurm_time` raised to match exp071's 14h (see config.yaml comment): exp130 measured 2B base-only
  generation at 4.6h, but applying exp071's own base-to-LoRA ratio (2.4x) projects close to a 10h
  cap, and LoRA overhead is not guaranteed to shrink with model size.

Everything else — 200 prompts, 10 classes, `erased_class: "chain saw"`, 50 inference steps,
`disable_mlflow` — is unchanged, so the row is comparable to exp071's under both the 1000-way and
restricted conventions, and to exp130's `Original` 2B row for the PSR delta.

## What to watch
- **Restricted ESR-1 / ESR-5 / PSR-1 / PSR-5** against GOAL.md's target table (92.38 / 77.09 / 54.03
  / 82.14) and its four guards, including the motion floor (0.15).
- **ESR-5 specifically**: T2VUnlearning's own claim is that baselines inflate ESR-1 while leaving
  ESR-5 low (distortion, not removal). exp071 showed our 5b row does this too under the restricted
  convention (ESR-1 49.0, ESR-5 10.0) despite passing outright under 1000-way (100.0 / 89.8). Watch
  for the same split here.
- **Per-class motion on the nine preserved classes**, against exp133's live-sample surprise (rising,
  not falling). This is the check exp133's notes explicitly deferred to this run.
- **Colorfulness and DOVER** on the erased class and the nine preserved ones, same over-saturation
  signal exp071 found on all ten 5b classes.

## Status
- [ ] Submitted.
- [ ] Row measured under both conventions; checked against GOAL.md's target table and all four
      guards (ESR-1, ESR-5, PSR-1, PSR-5, motion floor).
- [ ] Per-class motion on the nine preserved classes checked against exp133's live-sample reading.
