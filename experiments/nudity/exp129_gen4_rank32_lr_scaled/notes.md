---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  The valid version of exp125's capacity test: rank 32 at eta 4 with lr scaled to 2.5e-5 so the
  effective step matches rank-8, making direction count the only changed variable against exp124
  r1. Success = the same erasure window at higher DOVER-a (rank-8 ref 0.717 at s160); same
  distortion = capacity was not binding and eta4/rank-8 s160 stands. 1 job; a rank-128 arm only if
  this works.
---
# exp129 — rank 32, lr-scaled

Rationale and the exp125 post-mortem are in the config header and
[exp125's notes](../exp125_gen4_rank_sweep/notes.md).

## Bar
Two adjacent checkpoints at rate <=0.05 with DOVER-a >= 0.79 (the old incumbent's level) and clip
>= 0.28. DOVER decides — colour has now misled twice (exp124's "recovery" at 32 was saturation;
exp125's 49-75 was blowout).

## Status
- [ ] Submitted (1 job).
- [ ] Window compared to exp124 r1 at matched steps; DOVER-a on the window locally.
- [ ] If it wins: rank-128 arm decision, then full exp112 battery + human review.
