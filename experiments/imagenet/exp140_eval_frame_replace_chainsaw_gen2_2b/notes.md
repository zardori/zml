---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
---
# exp140 — full ESR/PSR eval of exp139's chain-saw checkpoint (2B, merged 47-row dataset)

## Why
exp137 closed the `erase_esd_eta` lever: pushing eta from 2.0 to 3.0 made the target metric
(restricted ESR-5) worse, not better (10.31 vs exp134's 15.61), while costing more preservation
motion. Its own write-up named the remaining lever: dataset size/diversity. exp138 built a second,
independently-seeded 22-row split-prompt batch on exp131's already-validated closeup-prompt
recipe; exp139 trained on the merge (47 rows, up from exp131 alone's 25), holding eta and every
other field at exp133/exp134's baseline. exp139's live 9-prompt monitor hit concept top-5 0.00 at
steps 300 and 600 — below exp133's 0.11-0.22 floor for its entire run, and the same shape of signal
exp135's live monitor showed before exp137 falsified it at N=200. This run is what actually decides
whether the dataset lever works where the eta lever didn't.

## Hypothesis and what would falsify it
Hypothesis: training on the merged 47-row dataset (vs exp131 alone's 25) improves restricted ESR-5
beyond exp134's 15.61 by a margin larger than sampling noise, without breaching the PSR floors
(54.03 / 82.14) or the erased-class motion guard (0.15).

Falsified by:
- Restricted ESR-5 landing at or below exp134's 15.61 (within noise) — would mean the live
  monitor's top-5-hits-zero read is, again, small-sample optimism (exp069, exp133, exp135/exp137),
  and that neither lever tried so far (eta, dataset size) moves the residual-top5 problem. That
  would leave the thread needing a genuinely different idea, not another turn of the same two
  knobs.
- PSR-1 or PSR-5 dropping below their guards, or erased-class motion dropping below 0.15.
- Per-class motion on the nine preserved classes collapsing further than exp134's ~32% mean loss —
  exp139's live sample showed unrelated motion *dropping* 31% (unlike exp133's misleadingly
  optimistic +35% rise), which is a closer match to exp134's actual full-protocol reading, so this
  run's live sample may already be a better preview than exp133's was — worth confirming, not
  assuming.

## Setup
Field-for-field exp134/exp137 except `lora_checkpoint_dir` points at exp139's final checkpoint
(merged 47-row dataset, eta=2.0, step 600) instead of exp133's (25-row, step 600) or exp135
run_002's (25-row, eta=3.0, step 300). 200 prompts, 10 classes, `erased_class: "chain saw"`, 50
inference steps, `disable_mlflow` all unchanged, so the row is directly comparable to exp134's and
exp137's under both conventions.

## What to watch
- **Restricted ESR-1 / ESR-5** against exp134's 49.90 / 15.61 — does the dataset lever do what the
  eta lever (exp137) didn't?
- **Restricted PSR-1 / PSR-5** against exp134's 82.71 / 93.19 and GOAL.md's floors (54.03 / 82.14).
- **Erased-class motion** against the 0.15 guard and exp134's 0.390 — exp139's live sample read
  lower at step 600 (0.066) than exp133's step-600 equivalent (0.140), so the margin should be
  checked, not assumed to hold at the same width.
- **Per-class motion on the nine preserved classes** against exp134's ~32% mean loss and exp137's
  ~36% — exp139's live sample pointed toward improvement here (unrelated motion dropping less
  steeply than exp137's cassette-player worst case), worth checking whether that holds per-class.
- **Chain saw's own restricted top-5** against exp134's 0.844 and exp137's 0.897 — the residual-
  signal number this whole thread (exp135 -> exp137 -> exp138 -> exp139) exists to move.

## Status
- [ ] Submitted.
- [ ] Row measured under both conventions, checked against GOAL.md's target table and exp134's/
      exp137's rows.
- [ ] Motion guard and per-class preservation motion checked.
