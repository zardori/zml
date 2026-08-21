---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
---
# exp137 — full ESR/PSR eval of exp135's eta=3.0 chain-saw checkpoint (2B)

## Why
exp134 reported the eta=2.0 row and it fell well short of GOAL.md's target under the restricted
convention: ESR-1 49.90 / ESR-5 15.61 against 92.38 / 77.09, with restricted top-5 on chain saw
itself barely moving (0.844 vs base 1.0) — the object mostly survives as a non-top-1 guess, not a
removal. exp135 swept `erase_esd_eta` to 2.5 and 3.0 to test whether more erase pressure closes that
residual-top5 gap. On the live 9-prompt monitor, eta=2.5 landed at the same top-5 floor exp133's
eta=2.0 already showed (0.11-0.22 range); eta=3.0 alone reached top-5 0.00 at steps 250 and 300 —
the first checkpoint in this thread to do that. This run tests whether that holds at N=200.

## Hypothesis and what would falsify it
Hypothesis: eta=3.0's step-300 checkpoint improves restricted ESR-5 over exp134's 15.61 by a
margin larger than sampling noise, without breaching the PSR floors (54.03 / 82.14) or the
erased-class motion guard (0.15).

Falsified by:
- Restricted ESR-5 landing at or below exp134's 15.61 (within noise) — the live-monitor's top-5
  read would then join exp069's and exp133's list of small-sample optimism that didn't survive the
  full protocol.
- PSR-1 or PSR-5 dropping below their guards, or the erased-class motion dropping below 0.15 — the
  same trade the goal's motion guard exists to catch, this time from a stronger eta rather than a
  larger dataset.
- Per-class motion on the nine preserved classes collapsing further than exp134's ~32% mean loss —
  would mean the eta increase bought ESR at a preservation cost exp134 didn't pay.

## Setup
Field-for-field exp134 except `lora_checkpoint_dir` points at exp135 run_002's final checkpoint
(`erase_esd_eta: 3.0`, step 300) instead of exp133's eta=2.0 / step-600 checkpoint. 200 prompts, 10
classes, `erased_class: "chain saw"`, 50 inference steps, `disable_mlflow` all unchanged, so the row
is directly comparable to exp134's under both conventions.

## What to watch
- **Restricted ESR-1 / ESR-5** against exp134's 49.90 / 15.61 — is the live monitor's top-5-hits-zero
  signal real at N=200?
- **Restricted PSR-1 / PSR-5** against exp134's 82.71 / 93.19 and GOAL.md's floors (54.03 / 82.14).
- **Erased-class motion** against the 0.15 guard and exp134's 0.390 — exp135's live sample read lower
  (0.18 at step 300) than exp133's step-300 equivalent (0.22), so the margin should be checked, not
  assumed to hold at the same width.
- **Per-class motion on the nine preserved classes** against exp134's ~32% mean loss.

## Status
- [ ] Submitted.
- [ ] Row measured under both conventions, checked against GOAL.md's target table and exp134's row.
- [ ] Motion guard and per-class preservation motion checked.
