---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  Tests the rank hypothesis from human review of exp124: eta-4's deep window (0.00-0.04 over steps
  60-160 at n=25) comes with visible distortion, and a rank-8 adapter can only implement a large
  displacement as a few coarse global directions — clothing and collateral ride the same vector.
  lora_rank [32, 128] at eta 4 on clean-75, alpha derived (= rank) via scripts/unlearn.py so the
  Cartesian grid cannot mismatch the pair. Receler precedent: their eraser is rank 128. 2 jobs.
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
