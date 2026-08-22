---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  THE DATASET LEVER ALSO FAILS. Restricted ESR-1 49.90 -> 52.55 (+2.65) and ESR-5 15.61 -> 15.82
  (+0.21) against exp134's rank-8/eta-2.0/25-row baseline — both movements are inside noise, and
  PSR-1 actually dropped (82.71 -> 81.34). Chain saw's own restricted top-5 barely moves (0.842 vs
  base 1.0), so the residual-signal problem — the object staying in the model's top-5 guess even
  after top-1 erasure — is untouched by nearly doubling the training set, the same way exp137 found
  it untouched by raising erase pressure. exp139's live-monitor read (top-5 hitting 0.00 at steps
  300 and 600, a level exp133's 25-row run never reached) is exactly the small-sample optimism
  exp135's live monitor showed before exp137 falsified it on the full protocol — it joins that list.
  Preserved-class motion (vs exp130's per-class base) lost a mean ~39% here (cassette player worst,
  -92%, matching exp137's -93% on the same class almost exactly; English springer rose +53%,
  matching the "outlier class always gains" pattern seen in exp133/exp139's live samples) — close to
  exp137's ~36% and worse than exp134's ~32%, so the merged dataset did not buy back any of the
  preservation exp137's higher eta cost, either. Erased-class motion guard still passes (0.262 vs
  the 0.15 floor) but with a smaller margin than exp134's 0.390 (base 0.840: -69% loss here vs -54%
  there). Two independent levers (eta, dataset size) now both null on the target metric — see
  exp141 for the next one.
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
- [x] Submitted. Completed on athena, 5.2h, exit 0 (job 3031483).
- [x] Row measured: restricted ESR-1 52.55 / ESR-5 15.82 / PSR-1 81.34 / PSR-5 93.30, against
      exp134's 49.90 / 15.61 / 82.71 / 93.19 and GOAL.md's target/guards (92.38 / 77.09 / 54.03 /
      82.14) — both ESR-1 and ESR-5 miss badly and by essentially the same margin as exp134 did;
      PSR-1/PSR-5 clear their floors with room to spare.
- [x] Motion guard checked: erased-class motion 0.262 vs the 0.15 floor (passes, base 0.840 per
      exp130). Per-class preservation motion checked against exp130's base: mean ~39% loss across
      the nine preserved classes (cassette player worst at -92%), slightly worse than exp134's ~32%.
