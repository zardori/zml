# exp022 — unhype fire, aggressive

`simulated_lr=0.01`, `retain_weight=0.1`. Largest trajectory step + relaxed retention;
strongest erasure, watch for collateral damage on unrelated/related control prompts. Part of
the exp020–022 single-run set; full rationale (incl. the dead-init fix that unblocks training)
in `experiments/exp021_unhype_fire_moderate/notes.md`.

## Outcome: failed (no change vs base)

Same two bugs as exp020/exp021, fixed in `exp023`:
1. **Constant-output init** (zero final-layer weight) → constant `B=0` no-op adapter (identical
   videos) and `θ(s+1)−θ(s)≡0` → removal loss trivially zero, nothing trained.
2. **`simulated_lr` ~40× too small** → endpoint `≈0.03`, inert anyway.
