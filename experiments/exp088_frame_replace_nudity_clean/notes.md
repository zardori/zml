---
status: active
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  NEGATIVE RESULT, and an important one: un-freezing the donors does NOT fix the motion collapse.
  run_001 (clean data, eta 1.5) tracks exp086 run_003 (old data, same eta) within 0.02-0.07 at every
  checkpoint and ends *lower* — 0.02 vs 0.08 against base 0.686. The frozen-donor diagnosis carried
  since exp080/exp087 is therefore wrong: the collapse is caused by the erase objective, not the
  data. The U-shape also survives intact (rate rebounds to 0.67 by step 160). run_002 (eta 2.0)
  still running and is the one live hope — 0.00 at step 130 where every other run had rebounded.
---
# exp088 — frame_replace on the clean dataset

## Why
exp080 is the same run on data where 20 of 34 targets were a single repeated donor frame. It
collapsed concept motion to **-87% .. -99%** against a base of 0.686 while unrelated motion was
untouched, and its best human-reviewed point (run_002 step 120) still cost **-85.2% motion** and
**-37.5% colorfulness**. Against exp083's NegPrompt baseline — 0.105 / 0.230 residual at *no*
measurable quality cost — that is not a checkpoint we can report.

exp087 fixed the data without regenerating it: re-edited from the saved originals, frozen targets
20/21 -> 0/21, edit/safe motion ratio 0.01 -> 1.00, and 7 detector-derived concept masks corrected
to construction. This run is exp080 again on that data.

## Unblocked (2026-08-10) — what exp085/exp086 decided

- **`erase_esd_eta: [1.5, 2.0]`.** Human review of both grids kept **exp080 run_002 (eta 2.0) as the
  best checkpoint to date**, and of exp086's three arms only eta 1.5 came close. Both values are
  carried deliberately: eta's benefit is partly "do not overfit a frozen donor", and that benefit
  shrinks once the donors are not frozen, so the optimum may move on this data and one point could
  not show it. It also gives a clean 2x2 against runs we already have on the old data at matched lr
  and retention — exp080 run_002 (eta 2.0) and exp086 run_003 (eta 1.5).
- **Retention stays exp041 (fire), unchanged.** exp085 settled that exp079's set is not the answer,
  and for a reason worth stating: its human-filtered 20 entries are **11/20 exposed-skin wardrobe**
  (swimwear x4, leotard, sports bra, pyjamas, towels x2, a bare-shoulders close-up, a midriff
  close-up), so retention was pulling toward keeping exposed torsos while the erase term pushed away
  from the same features. exp104 builds a fully-clothed replacement; **exp105** tests it on the
  **old** data so that arm isolates retention exactly as this one isolates data. Changing both here
  would confound them.

**This run answers one question: does un-freezing the donors fix the motion collapse?** Nothing else
about it differs from exp080 run_002.

## Setup
Identical to exp080 run_002 except the dataset, the two fields above, and the eval budget: concept
only at `save_interval: 10` (200 clips/run), matching exp085/exp086. exp080's good state was a
~40-step window that 20-step checkpoints straddled — at 5e-4 it opened and closed almost entirely
between two of them — so temporal resolution is where the budget goes.

## What to watch
- **Concept motion against base 0.686.** This is the whole point. exp080 landed at 0.03-0.16; if
  the clean data does not move that substantially toward 0.686, the frozen donors were not the
  cause and the diagnosis in exp080/exp087 is wrong.
- **Whether phase 4 survives.** exp080's human review found nudity returning at steps 140-200. If
  that was eta=2 overshooting a *frozen* target, it should weaken here. If it persists at the
  chosen eta, the instability is separate from the data problem.
- **Whether erasure still happens at all.** The frozen targets were a degenerate but very strong
  training signal. A mirrored fill is a subtler target, so it is possible erasure lands later or
  weaker — that would be a real trade, not a failure, and is what the 20 checkpoints are for.
- Per [[feedback-detector-metrics-not-ground-truth]], the four-phase structure was invisible to the
  metrics until someone watched the clips. n=10 detection rate ranks arms; it does not pick one.

## Results — run_001 (clean data, eta 1.5), complete

### The motion question, answered: no

This run existed to test one diagnosis. Against its matched control `exp086 run_003` — identical in
every field except the dataset — concept motion (base **0.686**):

| step | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|
| **exp088 r1** (clean) | 0.66 | 0.39 | 0.17 | 0.14 | 0.14 | 0.10 | 0.08 | 0.11 | 0.05 | **0.02** |
| exp086 r3 (old) | 0.67 | 0.45 | 0.23 | 0.16 | 0.15 | 0.17 | 0.11 | 0.11 | 0.11 | 0.08 |
| exp080 r2 (old, eta 2.0) | 0.67 | 0.41 | 0.15 | 0.14 | 0.14 | 0.11 | 0.09 | 0.11 | 0.04 | 0.03 |

The three curves are the same curve. The clean-data run is *below* its control at 8 of 10 shared
checkpoints and ends at 0.02 — a 97% loss. **Un-freezing the donors changed nothing about the motion
collapse**, so the diagnosis carried since exp080 and acted on in exp087 is refuted. Whatever
destroys motion is in the erase objective, not in the frozen targets.

exp107 independently corroborates this and locates it: the same LoRA costs **-68%** motion on VBench
`object_class` and **-36%** on `subject_consistency`, neither of which contains nudity. The collapse
is a global property of the adapter, which no change to the nudity training targets could have fixed.

### The U-shape also survives

| step | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 | 160 | 170 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frame rate | **0.00** | 0.11 | 0.26 | **0.00** | **0.00** | 0.20 | 0.11 | 0.40 | **0.67** | 0.53 | 0.30 |
| colour | 16.6 | 20.7 | 18.3 | 20.4 | 24.6 | 25.3 | 28.0 | 29.9 | 31.1 | 35.1 | 35.3 |

Erasure is still strongest where the model is most degraded and decays as colorfulness recovers,
exactly as in exp086 run_003 (which peaked at 0.76). The tail is somewhat lower than its control but
the shape is unchanged. Note the isolated zeros at 110/120 sitting between 0.26 and 0.20 — at n=10
(490 frames) a single zero is not a regime, and reading those two as one cost an incorrect conclusion
before the tail arrived.

## Results — run_002 (clean data, eta 2.0), through step 130

Still running. This arm looks materially different from r1:

| step | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 |
|---|---|---|---|---|---|---|---|---|---|
| rate | 0.13 | **0.00** | **0.00** | **0.00** | 0.01 | 0.01 | 0.10 | 0.10 | **0.00** |
| colour | 20.0 | 15.9 | 14.6 | 15.2 | 18.1 | 16.2 | 18.0 | 23.2 | 19.5 |

Five consecutive checkpoints at <=0.01 (steps 60-100), and **still 0.0000 at step 130** where exp088
r1 had rebounded to 0.20 and exp080 r2 reached 0.49 by 140. That width is the interesting claim — a
checkpoint chosen from a five-wide plateau is not selection-on-test in the way a single lucky step is.

Two honesty checks on it. exp080 run_002 was evaluated every **20** steps, so on the four shared
steps the record is 60 (0.00 vs 0.10), 80 (0.00 vs 0.00), 100 (0.01 vs 0.10), 120 (0.10 vs **0.00**)
— two wins, a tie, a loss. The plateau rests on steps 70 and 90, which the old-data run never
sampled, so it is suggestive rather than proven. And clip score drifts 0.29 -> 0.26 by step 130 here
while r1 held 0.28-0.29 throughout: eta 2.0 is eating text conditioning on this data.

## Downstream
The chosen checkpoint is what **exp084** should point at for the reported method row — that config
changes one field and nothing else, so the comparison against exp082/exp083 stays valid.

The refuted diagnosis changes what the remaining arms are for. **exp105** (clothed retention) is now
the only live hypothesis about the motion collapse: if the erase objective destroys motion, an anchor
set that pins "human, clothed, *moving*" is the mechanism most likely to hold it, and exp041's fire
anchors contain no people to pin. If exp105 also leaves motion at ~0.1, the collapse is intrinsic to
eta-extrapolated v-prediction erasure and belongs in the paper as a stated limitation rather than an
open bug.

## Status
- [x] exp087's dataset built and verified (motion ratio 1.00, 0 frozen).
- [x] `erase_esd_eta` and the retention set filled in from exp085/exp086.
- [x] Submitted, 2 jobs.
- [x] run_001 complete (200 steps); run_002 running (130/200 at 2026-08-10 22:30).
- [x] Concept motion compared against base 0.686 — **no improvement over exp080/exp086**.
- [ ] run_002 tail (140-200): does the step-130 zero hold, or does it rebound like every other arm?
- [ ] DOVER scored locally (helios wrote 0.0); needs the eval videos pulled.
- [ ] Best checkpoint chosen by human review; exp084 re-pointed at it.
