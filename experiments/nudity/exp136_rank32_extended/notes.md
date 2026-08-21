---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  exp129 at 400 steps — both rank-32 arms were truncated mid-story at 200 by the (deliberately
  conservative) lr scaling. The eta-2 arm ended at its best point still descending: 0.08 at colour
  31 / clip 0.28, the same rate rank-8 bottomed at but +9 colour and no rebound in sight — the
  capacity-substitutes-for-push signal. The eta-4 arm ended mid-window with colour still
  recovering. Fresh 400-step run, shared range doubles as reproducibility check. 2 jobs.
---
# exp136 — rank 32, 400 steps

Rationale in the config header. The 2x2's rank-8 cells are exp123 r1 / exp124 r1; exp129 holds the
truncated rank-32 curves this extends.

## What decides it
- **eta 2 arm**: does the descent continue below 0.05 while DOVER-a holds near incumbent level?
  Passing the bar here means erasure without the degenerate trough at all.
- **eta 4 arm**: does the window survive into recovered colour (rank-8 pattern), and at what
  DOVER-a vs rank-8's 0.72?
- Steps 0-200 must reproduce exp129 within noise, else neither run is trustworthy.

## Status
- [ ] Submitted (2 jobs).
- [ ] 0-200 overlay vs exp129 (reproducibility).
- [ ] DOVER on candidate windows; bar applied.
- [ ] Winner through the full exp112 battery + human review.
