---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  exp080's regime on exp087's re-edited dataset — isolates the DATA variable. Unblocked 2026-08-10:
  eta [1.5, 2.0], retention held at exp041 (fire). Read against exp080 run_002 and exp086 run_003,
  which are the same two eta points on the old data. Ready to submit, 2 jobs.
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

## Downstream
The chosen checkpoint is what **exp084** should point at for the reported method row — that config
changes one field and nothing else, so the comparison against exp082/exp083 stays valid.

## Status
- [x] exp087's dataset built and verified (motion ratio 1.00, 0 frozen).
- [x] Config staged; dataset path final, two fields pending.
- [ ] `erase_esd_eta` and the retention set filled in from exp085/exp086.
- [ ] Submitted.
- [ ] Concept motion compared against base 0.686 and against exp080's 0.03-0.16.
- [ ] Best checkpoint chosen by human review; exp084 re-pointed at it.
