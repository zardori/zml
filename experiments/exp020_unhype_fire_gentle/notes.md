# exp020 — unhype fire, gentle (preservation-first)

`simulated_lr=0.002`, `retain_weight=1.0`. Smallest trajectory step + strongest retention.
Part of the exp020–022 single-run set; full rationale (incl. the dead-init fix that unblocks
training) in `experiments/exp021_unhype_fire_moderate/notes.md`.

## Outcome: failed (no change vs base)

Two compounding bugs, both fixed in `exp023`:
1. **Constant-output init.** The hypernet's final-layer weight was zero-init'd, so its output
   was a constant (the bias) for every `(c, s)`: `B=0` ⇒ no-op adapter (identical videos), and
   `θ(s+1)−θ(s)≡0` ⇒ removal loss trivially won by the zero trajectory (`loss≈3.7e-18`). Fixed
   in `unhype_modules.py` (small nonzero weight, B-rows zeroed so output varies with `(c,s)`).
2. **`simulated_lr` ~40–200× too small.** Endpoint `≈ S·η·‖∇ℒ‖ ≈ 0.03`, far below a
   generation-affecting adapter — inert even without bug 1.
