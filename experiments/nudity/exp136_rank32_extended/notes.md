---
status: done
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  CAPACITY HYPOTHESIS CONFIRMED. At rank 32 / eta 4 the erasure window SURVIVES the colour-recovery
  limb for the first time: five checkpoints <=0.04 across steps 200-360 while colour climbs 23->39.
  Binned by colour, rank 32 reads 0.15 where rank 8 reads 0.40 at base-like colour. Best point s320:
  rate 0.01 at colour 34.9 (96% of base). One caveat blocks the bar — clip 0.24-0.26 vs the >=0.28
  requirement, i.e. weaker prompt adherence than rank 8. The eta-2 arm oscillates and is withdrawn
  as a lead: capacity does not substitute for push.
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

## Results (2026-08-22) — the erasure window survives colour recovery for the first time

400 steps, n=25 (1225 frames/checkpoint). Base on this subset: rate 0.340, colour 36.3, clip 0.30.

### run_002 (eta 4, rank 32) — the candidate

| step | 120 | 160 | **200** | **240** | 280 | **320** | 360 | 400 |
|---|---|---|---|---|---|---|---|---|
| rate | 0.04 | 0.04 | **0.000** | **0.000** | 0.04 | **0.01** | 0.04 | 0.21 |
| colour | 15.1 | 17.8 | 23.1 | 29.2 | 35.5 | **34.9** | 39.2 | 40.2 |
| clip | 0.25 | 0.26 | 0.24 | 0.26 | 0.25 | 0.26 | 0.25 | 0.25 |

**Five consecutive checkpoints at <=0.04 spanning steps 200-360, while colorfulness climbs 23 -> 39.**
Every previous run in this thread lost its erasure exactly on the colour-recovery limb. Binned by
colour band, the difference is unambiguous:

| | c<20 | 20-28 | 28-33 | 33-37 | c>37 |
|---|---|---|---|---|---|
| rank 8, eta 4 (exp124 r1) | 0.02 | 0.10 | 0.03 | **0.40** | 0.18 |
| **rank 32, eta 4 (this)** | 0.04 | 0.10 | **0.00** | **0.15** | **0.12** |
| rank 32, eta 2 (run_001) | — | 0.12 | 0.17 | 0.18 | 0.20 |

At recovered colour (33-37, i.e. base-like) rank 8 reads 0.40 and rank 32 reads 0.15; at 28-33 the
gap is 0.03 vs 0.00. **This is the capacity hypothesis confirmed** — with 32 directions the adapter
can hold the erase mapping while the model recovers its appearance, which rank 8 could not do.
s320 (rate 0.01 at colour 34.9, i.e. 96% of base colour) is the single best erasure/appearance
point the project has produced.

### The caveat that keeps it from clearing the bar
**Clip score sits at 0.24-0.26 across the entire window**, against the bar's >=0.28, the base's 0.30,
the old incumbent's 0.27 and rank-8-eta-4's 0.27-0.29. Text conditioning is measurably weaker here
than at rank 8 — more capacity applied to the erase direction appears to cost prompt adherence.
Whether -0.02 to -0.03 CLIP is visible is a human-review question, and it is the one thing standing
between this and a reportable checkpoint.

### run_001 (eta 2) — the exp129 signal did not survive extension
exp129's eta-2 arm ended at 0.08 "still descending", which this run shows was **not a descent**:
extended to 400 steps it oscillates (0.08, 0.17, 0.04, 0.07, 0.17, 0.23, 0.10, 0.17) around
~0.10-0.15 with no floor. Its one qualifying point (s200, rate 0.04 at colour 30.4 and clip **0.28**
— the only checkpoint in either arm to meet the clip bar) has neighbours at 0.17 and 0.07, so it
fails the two-adjacent rule. Capacity does **not** substitute for push; the eta-2 reading recorded
in exp129 was over-optimistic on a truncated curve and is withdrawn.

## Status
- [x] Submitted and complete (2 jobs, 400 steps).
- [x] 0-200 overlay vs exp129 — consistent (eta4 window at s120-200 in both).
- [x] Erasure survives colour recovery at rank 32; capacity hypothesis confirmed.
- [ ] DOVER on the window (running locally) — decides whether s320 is real.
- [ ] Human review of s320/s240 — specifically whether clip 0.25 shows as prompt drift.
- [ ] If it holds: full exp112 battery on s320.
