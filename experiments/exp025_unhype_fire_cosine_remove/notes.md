# exp025 — unhype fire, cosine removal loss (decouple direction from magnitude)

Single run, **no grid**. Lever #2 of two follow-ups to the exp023 diagnosis.

## Why exp023 failed (recap)

Generations never changed: `eval.concept.fire_detection_rate` pinned at 1.0, CLIP scores
byte-identical across steps 500/1000/1500. The endpoint adapter `θ(S)` stayed a **no-op**.

Root cause: `loss_remove = MSE(predicted_step, target_step)` was dominated by `‖predicted_step‖²`
(`predicted_step` ~0.13–0.36 vs `target_step` ~0.002, a 120× gap), so its gradient shrank the
predicted step toward zero → `θ(s)` collapsed to constant-in-`s` → endpoint stayed at `θ(0)` →
no-op adapter.

## What changed vs exp023

- **New removal loss form** (config `remove_loss_type: cosine`, in `zml/unlearn/unhype.py`):

  `loss_remove = (1 − cos(pred, target)) + remove_magnitude_weight·(‖pred‖ − ‖target‖)²`

  The **direction** term aligns the predicted step with the erasure gradient *without* a term that
  drives its magnitude to zero. The **magnitude** term (`remove_magnitude_weight: 1.0`) is a
  subordinate soft anchor.
- **`simulated_lr` stays 0.3** — this run isolates the loss-form change from the magnitude lever
  tested in exp024.
- `steps` 2000 → 1000, `slurm_time` 1d → 12h. Checkpoints at 500/1000 for resume.

## Success criteria (in order)

1. `train/loss_remove_direction` falls toward 0 (the step is aligning with the erasure gradient).
2. `train/predicted_step_norm` holds meaningfully above ~0.1 instead of decaying toward 0
   (no collapse to a constant trajectory).
3. `eval/theta_S_norm_concept` moves off 15.16 with real spread across prompts.
4. `concept.fire_detection_rate` drops while `unrelated`/`related` stay clean.

## Notes / caveats

- With `cosine`, `train/loss_remove` is now O(0.1–1), **not** ~1e-9 as in exp023. That is expected
  and not a regression; the `loss_remove_degenerate` health flag keys off ~1e-18 and won't trip.
- If `predicted_step_norm` still collapses, the magnitude anchor is too strong — drop
  `remove_magnitude_weight` toward 0 next iteration.
