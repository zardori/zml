---
status: done
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  UNINFORMATIVE BY DESIGN ERROR — the rank hypothesis remains untested. Pairing alpha = rank kept
  the per-direction scale at 1.0, so the adapter's effective step size scaled with rank at fixed
  lr: rank 128 hit its trough by step 20 (colour 7.5) and thrashed for the rest of the run (colour
  50-75, rate oscillating 0.04-0.43); rank 32 oscillated with no stable window and clip never
  above 0.27. Neither arm met the bar or produced a candidate. The correct control was constant
  effective step — lr scaled inversely with rank — which is exp129. 2 jobs spent, lesson recorded.
---
# exp125 — rank [32, 128] at eta 4

Full rationale in the config header. The one-line version: exp124-eta4 s160 already dominates the
old incumbent on paper (rate 0.030 vs 0.120 first-25, colour 32 vs 24, motion/clip equal), but
human review reads the window as distorted; if that distortion is a capacity artifact, rank fixes
it, and if it persists at rank 128 the extrapolation itself is the limit.

## Bar
Hold rate <=0.05 across two adjacent checkpoints at clip >=0.28 and colour >=30 — exp124's window
without the distortion. Winner gets the full exp112 battery + human review.

## What to watch
- **DOVER-aesthetic in the window** — the instrument that separates "clean but static" from
  "distorted"; exp124's window values are being scored locally now and are the reference.
- Rebound timing: more capacity can memorize the 75 targets faster; the tail is not the read.
- clip score at rank 128 — a 16x adapter at lr 1e-4 may move faster than the eval grid resolves.

## Status
- [ ] Submitted (2 jobs).
- [ ] Window quality vs exp124-eta4 at matched steps (rate + colour + clip + DOVER-a).
- [ ] Winner through full battery + human review.


## Results (2026-08-19) — uninformative, and why

| | rank 32 | rank 128 |
|---|---|---|
| trough arrives | step 40 (colour 9.9) | **step 20** (colour 7.5) |
| stable window | none — rate oscillates 0.00-0.21 | none — 0.00 -> 0.43 -> 0.04 -> 0.26 |
| colour range after trough | 20-49 | **48-75** (base 36.3) |
| max clip in any low-rate checkpoint | 0.27 | 0.29 (at rate 0.43) |

Neither arm holds two adjacent checkpoints at rate <=0.05 with acceptable quality. This is not
evidence against the rank hypothesis — it is a broken experiment:

**The design error.** alpha was derived as alpha = rank ("the thread convention", but that
convention only ever existed at rank 8). With scale = alpha/rank fixed at 1.0, every added rank
direction receives gradient and adds to the output, so the adapter's effective per-step
displacement grows roughly linearly with rank at fixed lr. Rank 128 therefore trained ~16x too
fast: its entire usable window fell before the first checkpoint (step 20 already at colour 7.5),
and everything after is overshoot thrash — the colour blowouts to 75 are the exp124 style-direction
artifact amplified by capacity. Rank 32 shows the same failure at 4x speed.

**What a valid capacity test needs:** constant effective step across ranks — lr scaled inversely
with rank (rank 32 @ 2.5e-5, rank 128 @ 6.25e-6) with alpha = rank retained. That is
[exp129](../exp129_gen4_rank32_lr_scaled/notes.md), run first at rank 32 only: one job, and if the
window reappears with better DOVER-a than rank-8's 0.717, the capacity story is live and a 128 arm
follows.

DOVER was deliberately not scored on these outputs: no checkpoint is a candidate, and the metrics
alone disqualify every point.

## Status
- [x] Submitted and complete (2 jobs).
- [x] Read: no stable window in either arm; design error identified (alpha=rank scales step with rank).
- [ ] Superseded by exp129 (lr-scaled rank test).
