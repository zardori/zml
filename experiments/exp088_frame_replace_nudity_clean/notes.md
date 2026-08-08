---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  exp080's regime on exp087's re-edited dataset — the candidate for the reported checkpoint. Two
  fields (erase_esd_eta, retention set) are placeholders until exp085/exp086 report. Blocked on
  those, not on compute.
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

## Blocked on, and the decision rule
Two fields are placeholders. Both are answered by grids that are running now:

- **`erase_esd_eta`** <- whichever value exp086 picks (exp085 should agree; if they disagree, that
  itself is a finding about how retention interacts with the overshoot). The config carries **two**
  values deliberately: eta<1's benefit in exp085/exp086 is partly "do not overfit a frozen donor",
  and that benefit shrinks once the donors are not frozen — so the optimum can move on this data and
  should not be assumed to transfer. Two jobs is cheap insurance against re-running the grid.
- **`retention_metadata_file` / `retention_latents_dir`** <- exp041's fire-era anchors or exp079's
  nudity anchors, whichever exp085-vs-exp086 favours at matched eta. They must be changed together.

They sit at exp080's values so the file is runnable, **not** because those are the answer.

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
