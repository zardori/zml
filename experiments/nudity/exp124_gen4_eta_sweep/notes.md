---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  eta [4.0, 7.0] on the CLEAN-75 dataset, completing a four-point eta curve (2, 3, 4, 7) with
  exp123's arms — one dataset, one eval, one seed. The "clean-75 hurt" reading is corrected: it was
  a phase confound (75 targets = 4/3 the visits per step; at matched visits 75 and 100 read 0.14 vs
  0.123). eta 7 targets the OLD recipe's effective push: eta*(donor gap), and fitted donors have
  roughly a third the gap of the old sacks. Watch clip score on the 7.0 arm. 2 jobs.
---
# exp124 — eta [4, 7] on clean-75

Rationale, the phase-confound correction, the eta-7 scaling argument, risks, and the bar are all in
the config header. Summary of the curve this completes, all on clean-75 at n=25:

| eta | run | s140 rate | source |
|---|---|---|---|
| 2.0 | exp123 r1 | 0.26 | done |
| 3.0 | exp123 r2 | 0.11 | done |
| 4.0 | this, run_001 | ? | |
| 7.0 | this, run_002 | ? | |

References on the same first-25 subset: old ckpt 0.1200, gen4-100 ckpt 0.1233, base ~0.41.

## What to watch
- **clip score on eta 7** — conditioning collapse is the failure mode that ends the eta line.
- The trough may shift left (deeper push, earlier overfit); interval-20 sampling of an early narrow
  window is the known resolution risk.
- Two adjacent checkpoints, never one.
- Winner gets the full exp112 battery + human review before any claim.

## Status
- [ ] Submitted (2 jobs).
- [ ] Four-point eta curve assembled against exp123.
- [ ] Winner (if any) through full battery + human review.
