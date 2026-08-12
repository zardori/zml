---
status: ready
concept: nudity
method: eval
thread: nudity
takeaway: >
  Base + NegPrompt on `prompts/ring_a_bell_nudity_safe.csv` (79 safe rewrites at matched seeds).
  exp102 measured OUR checkpoint at 0.0000 there, but with no baselines it is a cell rather than a
  column and cannot be reported. T2VUnlearning publishes no nudity preservation column at all, so
  this is a place to be strictly more complete than the comparison. 2 jobs.
---
# exp111 — baselines on the safe (related) set

## Why

`prompts/ring_a_bell_nudity_safe.csv` is 79 safe rewrites of the Ring-A-Bell nudity prompts at
**matched seeds** — same scene, same seed, clothed subject. It is the cleanest preservation probe we
have, because the only thing that differs from the erased distribution is the concept itself.

[exp102](../exp102_eval_frame_replace_comparable_nudity/notes.md) scored our checkpoint on it and got
**0.0000**. That number is currently unreportable: base and NegPrompt were never run on the set, so
there is nothing to compare against and no way to know what the detector's floor is on these prompts.
Given that base reads 0.043-0.053 on VBench people-prompts
([exp106](../exp106_vbench_utility_base/notes.md)), the floor here is probably non-trivial — and if
base reads, say, 0.05 on the safe set, then our 0.0000 is *below* the floor and that is a stronger
statement than "we score zero".

**T2VUnlearning reports no nudity preservation column of any kind** — only VBench utility, and only
on HunyuanVideo. So this is not catching up; it is a column we can have and they cannot.

## Design
Two arms in one config so prompts, seeds and inference settings are provably identical:

- `run_001` — empty negative prompt, i.e. the base model. Establishes the detector floor.
- `run_002` — the exact NegPrompt string used in exp083 / exp101 / exp102, so this row stacks with
  every other NegPrompt number in the comparable table.

The set is filed under `control_concept_prompts` because that is the slot scored by the concept
detector; it appears as **related** in the results table. **A low number here is the good outcome** —
these prompts contain no nudity, so this measures whether a method destroys adjacent-but-safe content.

## What to watch
- **The base row is the point.** It is the floor against which our 0.0000 and NegPrompt's number are
  read. Without it neither is interpretable.
- **Motion and DOVER on this set matter as much as the rate.** exp102's related motion reads 0.04 for
  our checkpoint; if base reads ~0.7 there, the freezing extends to safe adjacent content and that is
  a preservation failure the rate column alone would hide.
- DOVER will be 0.0 from helios (aarch64) — score locally with `tools/score_dover.py`.

## Status
- [ ] Submitted (2 jobs).
- [ ] Base floor recorded; our 0.0000 and NegPrompt read against it.
- [ ] DOVER + motion compared against exp102's related row.
