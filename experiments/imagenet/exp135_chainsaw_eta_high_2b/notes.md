---
status: ready
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  Not yet run.
---
# exp135 — erase_esd_eta sweep above 2.0, chain saw on CogVideoX-2B

## Why
exp134 landed the reported 2B chain-saw row and it falls well short of GOAL.md's target under the
convention that counts (restricted, 10-way): ESR-1 49.90 / ESR-5 15.61 against 92.38 / 77.09. PSR
clears its guards comfortably (PSR-1 82.71 vs 54.03 floor, PSR-5 93.19 vs 82.14 floor) and so does
the erased-class motion guard (0.390 vs the 0.15 floor; base was 0.840, exp130). Restricted top-5 on
chain saw itself barely moves (0.844 vs base 1.0) — the object mostly survives as a non-top-1 guess,
i.e. this reads as residual signal surviving erasure, not full removal.

Two things point at `erase_esd_eta` specifically as the lever with headroom left:
1. It has only ever been swept up to 2.0 (exp126, on 5b): 1.0/1.5 oscillated, 2.0 was the only arm
   that held top-1 flat at every checkpoint, so 2.0 became the fixed value carried into every run
   since, including exp133/exp134 — not because higher values were tried and rejected.
2. exp134's motion numbers show real margin before the guard: chain saw's own motion is 0.390
   (2.6x the 0.15 floor), and the nine preserved classes lost a mean ~32% of their motion against
   exp130's base numbers (computed here, not in exp134's report) — real but well short of exp071's
   5b finding of a ~45% mean loss, so there is room to trade some of that margin for more erase
   pressure before hitting the guard 5b already lives close to.

## Hypothesis and what would falsify it
Hypothesis: raising `erase_esd_eta` past 2.0 (to 2.5 and 3.0) pushes the restricted ESR-1/ESR-5
closer to GOAL.md's target by reducing the residual top-5 signal exp134 found, at some cost to
preserved-class motion that stays inside the 0.15 floor at these two values.

Falsified by any of:
- **No dose-response**: 2.5 and 3.0 land at materially the same erased-class top-1 as exp133's
  eta=2.0 arm on the *live* 9-prompt monitor (i.e. the pressure knob is already saturated at 2.0,
  the way exp126 found the tail-conditioning knob was inert in exp119). That would mean the ceiling
  exp134 measured is not an eta effect and the next lever to try is dataset size or diversity, not
  this one.
- **Motion floor breached early**: erased-class motion on the live monitor drops toward exp069/exp126's
  5b freeze range (0.01-0.05) before top-1 visibly improves — the same failure mode eta was already
  shown not to fix on 5b (exp126), just reached from a different eta this time.
- **Oscillation returns**: either arm's live top-1 swings back up mid-run the way 5b's eta 1.0/1.5
  arms did — would mean pushing eta *up* from 2.0 costs the stability 2.0 bought, not just the
  motion margin.

Only a live-monitor read is planned for this run (matching exp133's own gating step before its full
eval, exp134). A full 200-prompt `esr_psr` pass is deliberately *not* queued in this turn — if the
live monitor shows no dose-response or an early motion floor breach, the full eval would be spent on
a checkpoint already known not to beat exp134's row.

## Setup
Field-for-field exp133 except:
- `erase_esd_eta: [2.5, 3.0]` in place of the fixed `2`. 2.0 itself is not repeated as a third arm —
  it is already measured (exp133/exp134), so re-running it would burn budget on a known point.
- `steps: 300`, `save_interval: 50` in place of 600/100 — exp133's live monitor was already flat
  (top-1 0.00) from step 200 through step 600, so the back half measured nothing; a stronger eta's
  interesting window, if there is one, is early. Same reasoning and same budget exp126 used for its
  own eta sweep on 5b.
- `slurm_time: "0-06:00:00"` — half of exp133's step count at proportional headroom to its 3.7h/12h
  actual.

Dataset (`metadata_file`/`latents_dir`, exp131), retention set (exp132), prompts, LoRA
rank/alpha/dropout, learning rate and eval cadence are unchanged, so any difference from exp133's
eta=2.0 trajectory is attributable to the eta value alone.

## What to watch
- **Erased-class top-1/top-5 on the live 9-prompt set**, against exp133's eta=2.0 trajectory
  (0.09→0.00 top-1 by step 200; top-5 settling 0.11-0.22) — does a higher eta reach the same floor
  faster, or push top-5 lower than 2.0 did?
- **Erased-class motion on the live set** — exp133's eta=2.0 arm read 0.339→0.140 (step 100→600).
  Watch for it heading toward 5b's 0.01-0.05 freeze range before step 300 is reached.
- **Unrelated-set motion on the live set** — exp133's read here (rising 0.349→0.473) turned out to
  be a small-sample artifact once exp134's full protocol showed the nine preserved classes actually
  losing motion (~32% mean). Do not read this run's live signal as settling the question either;
  it is a gate for whether the full eval is worth spending, not the reported number.
- **Whether 2.5 and 3.0 move together or diverge** — a monotonic trend argues for chasing eta
  further; both landing at the same place as each other (and as 2.0) argues the knob is saturated.

## Status
- [ ] Submitted.
- [ ] Live-monitor trajectories checked against exp133's eta=2.0 run.
- [ ] Decision made on whether either arm's checkpoint is worth the full 200-prompt `esr_psr` eval.
