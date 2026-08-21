---
status: done
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  The lr control WORKED (no exp125 thrash, clip 0.28-0.30 through most of both runs) but linear
  lr-in-rank scaling overcorrected the pace and 200 steps truncated both arms mid-story. eta-2
  ended at its best point STILL DESCENDING: 0.08 at colour 31.0 / clip 0.28 — rank-8's bottom rate
  at +9 colour, no rebound in sight, the capacity-substitutes-for-push signal cut off where it got
  interesting. eta-4 reproduced the degenerate trough at shifted phase (window s100-200, colour
  13-23 still recovering at the end). No bar verdict possible; exp136 extends both to 400 steps.
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


## Results (2026-08-21) — truncated, but the eta-2 signal is real

| | rank 8 (exp123 r1 / exp124 r1) | rank 32 (this) |
|---|---|---|
| eta 2 trough | 0.07-0.08 @ colour 21-23 (s60-80), rebounds to 0.26+ | **0.08 @ colour 31.0, clip 0.28 (s200) — still descending, no rebound** |
| eta 4 window | s60-160, <=0.04, DOVER-a 0.72 at exit | s100-200, <=0.04, colour 13-23 **still recovering at horizon** |

The pace ran ~1.5-2x slower than rank-8 rather than matched — linear-in-rank lr scaling
overcorrects (the honest scaling likely sits nearer sqrt(rank)). Consequence: smooth, stable arms
that both hit step 200 before their decision points.

**The eta-2 arm is the result worth extending.** At the same rate where rank-8 bottomed out and
turned around, rank 32 sits 9 colour points higher with clip 0.28 and the curve still pointing
down. If that continues below 0.05 at held DOVER, it is erasure with no degenerate trough at all —
the mechanism-C prediction (capacity lets garment-specific directions replace the coarse shared
push). exp136 answers it at 400 steps.

DOVER scoring of both arms ran locally after the pull; numbers land in the metrics files.

## Status
- [x] Submitted and complete (2 jobs, 200 steps).
- [x] Read against the rank-8 cells; both arms truncated before the bar could be applied.
- [ ] Superseded operationally by [exp136](../exp136_rank32_extended/notes.md) (400 steps).
