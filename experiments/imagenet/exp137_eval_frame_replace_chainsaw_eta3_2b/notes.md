---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  FALSIFIED, BY ITS OWN PRE-REGISTERED CRITERION. Restricted ESR-5 landed at 10.31 — below exp134's
  eta=2.0 baseline of 15.61, not above it — so the live 9-prompt monitor's top-5-hits-zero signal
  (exp135) did not survive the full 200-prompt protocol; it joins exp069's and exp133's small-sample
  optimism failures. ESR-1 moved only 49.90 -> 53.57 (+3.7 points against the 92.38 target).
  Preservation held above its floors (PSR-1 81.03, PSR-5 91.55 vs 54.03/82.14) and the erased-class
  motion guard passed (chain saw 0.371 vs the 0.15 floor, base 0.840) — but preservation cost more
  than eta=2.0 paid: mean motion loss on the nine preserved classes (vs exp130's base) is ~36% here
  against exp134's ~32%, and cassette player specifically dropped to 0.039 (base 0.560, -93%),
  edging into 5b's frozen-poster range (0.01-0.05) though not below the guard, which is defined only
  on the erased class. Net: pushing erase_esd_eta past 2.0 buys a few ESR-1 points at a growing
  preservation cost while making the actual target metric (ESR-5) worse — the eta lever is
  exhausted. exp138 tests the alternative exp135 already named: dataset size/diversity.
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

## Results (2026-08-22)

Completed on athena, job 3028245, 5.5h elapsed.

Restricted (10-way) convention, eta=3.0/step-300 vs exp134's eta=2.0/step-600 vs GOAL.md's bar:

| metric | eta=2.0 (exp134) | eta=3.0 (this run) | target/guard |
|---|---|---|---|
| ESR-1 | 49.90 | 53.57 | 92.38 |
| ESR-5 | 15.61 | **10.31** | 77.09 |
| PSR-1 | 82.71 | 81.03 | 54.03 (floor) |
| PSR-5 | 93.19 | 91.55 | 82.14 (floor) |
| erased-class motion | 0.390 | 0.371 | 0.15 (floor) |

Chain saw restricted top-5 accuracy is 0.897 here (vs exp134's 0.844) — the object is *more* likely
to still show up in the top 5 at eta=3.0 than at eta=2.0, the opposite of what exp135's live monitor
suggested (top-5 hit 0.00 at steps 250/300 on 9 prompts).

Checked against the three pre-registered falsification conditions:
- **"Restricted ESR-5 landing at or below exp134's 15.61 (within noise)"** — MET, and not marginally:
  10.31 is below 15.61 by more than plausible sampling noise on 20 videos/class. The live-monitor
  read does not hold at N=200.
- **PSR floors** — not breached (81.03/91.55 both well above 54.03/82.14).
- **Preserved-class motion collapsing further than exp134's ~32% mean loss** — met, mildly. Computed
  against exp130's per-class base `motion_score_mean`: mean loss across the nine preserved classes is
  ~36% here (worst: cassette player -93% to 0.039, French horn -70% to 0.324, garbage truck -48% to
  0.360; best: English springer -1.8%, tench -3.3%). Cassette player's 0.039 sits inside 5b's
  frozen-poster range (0.01-0.05, exp069/exp126) even though the *erased*-class motion guard — the
  only one GOAL.md gates on — still passes at 0.371.

**Verdict**: exp135's hypothesis is falsified. Raising `erase_esd_eta` past 2.0 does not close the
ESR-5 gap; on the full protocol it makes ESR-5 worse while costing more preserved-class motion. 2.0
remains the best eta measured on 2B (exp134's row). Do not sweep eta further on this class without a
new reason. exp138 (a second-generation split-prompt build, same recipe as exp131 with fresh seeds)
opens the alternative lever exp135 itself named: dataset size/diversity.

## Status
- [x] Submitted.
- [x] Row measured under both conventions, checked against GOAL.md's target table and exp134's row.
- [x] Motion guard and per-class preservation motion checked.
