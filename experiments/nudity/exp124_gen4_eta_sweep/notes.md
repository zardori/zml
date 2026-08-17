---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  eta [3.0, 4.0] on the full gen4-100. exp123 showed the erase push scales as eta*(donor-teacher)
  — eta 3 beat eta 2 everywhere on identical data — and that removing the detector-visible targets
  HURT, so the full 100 is the right base. Bar on the shared first-25 subset: beat 0.12 at colour
  >=30. run_001 also gives a free 75-vs-100 read at eta 3 against exp123 r2. 2 jobs.
---
# exp124 — eta sweep on full gen4

See config header for the full rationale. In brief: gen4's fitted donors sit close to the nude
teacher, so eta=2 (tuned on baggy-sack donor gaps) under-pushes; exp123 r2 showed eta 3 recovers
erasure monotonically at every step. This tests 3.0 and 4.0 on the full 100-target dataset that
r1 showed is stronger than the clean subset.

## Bar
On the first-25 Gen subset (all numbers same instrument, same prompts): old ckpt 0.1200, gen4-100
ckpt 0.1233, exp123 r2 s140 0.1100. A winner reads below ~0.10 at colour >=30 and clip >=0.28,
across two adjacent checkpoints, and then survives the full exp112 battery.

## What to watch
- clip score on the eta 4 arm — extrapolation strength is where conditioning historically slips.
- Two adjacent checkpoints, never one (the n=25 version of the standing rule).
- The eta-3 arm vs exp123 r2 at matched steps: a free second read on clean-75 vs full-100.

## Status
- [ ] Submitted (2 jobs).
- [ ] Read against exp123 and the calibrated baselines.
- [ ] Winner (if any) through the full exp112 battery + human review.
