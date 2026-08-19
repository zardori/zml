---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  The valid version of exp125's capacity test, extended to a rank x eta 2x2: rank 32 at eta
  [2.0, 4.0], lr scaled to 2.5e-5 so effective step matches rank-8. The rank-8 cells exist
  (exp123 r1, exp124 r1), so each comparison is one variable. eta-4 arm asks whether capacity
  fixes the distortion; eta-2 arm asks whether capacity SUBSTITUTES for push — erasure without
  the trough at all, the best possible outcome. DOVER-led bar. 2 jobs.
---
# exp129 — rank 32, lr-scaled

Rationale and the exp125 post-mortem are in the config header and
[exp125's notes](../exp125_gen4_rank_sweep/notes.md).

## The 2x2

| | eta 2.0 | eta 4.0 |
|---|---|---|
| rank 8 | exp123 r1: 0.26 @ s140, mild quality cost | exp124 r1: 0.00-0.04 window, DOVER-a 0.72 |
| rank 32 | **run_001** — capacity as substitute for push? | **run_002** — capacity as distortion fix? |

## Bar
Two adjacent checkpoints at rate <=0.05 with DOVER-a >= 0.79 (the old incumbent's level) and clip
>= 0.28. DOVER decides — colour has now misled twice (exp124's "recovery" at 32 was saturation;
exp125's 49-75 was blowout). For the eta-2 arm specifically, even partial erasure recovery
(0.26 -> ~0.10) at near-clean DOVER would be the more interesting result than the eta-4 arm
winning.

## Status
- [ ] Submitted (1 job).
- [ ] Window compared to exp124 r1 at matched steps; DOVER-a on the window locally.
- [ ] If it wins: rank-128 arm decision, then full exp112 battery + human review.
