---
status: active
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

## Results (2026-08-17)

### eta 4.0 — a 100-step window at <=0.04, including two exact zeros at n=25

| step | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|
| rate | **0.000** | 0.04 | 0.03 | 0.04 | **0.000** | 0.03 | 0.40 | 0.02 |
| colour | 14.3 | 13.3 | 18.9 | 24.0 | 25.6 | 32.0 | 35.2 | 53.5 |
| motion | 0.02 | 0.02 | 0.05 | 0.08 | 0.05 | 0.05 | 0.08 | 0.04 |
| clip | 0.25 | 0.25 | 0.25 | 0.29 | 0.28 | 0.27 | 0.28 | 0.26 |

Steps 60-160 sit at 0.00-0.04 — five consecutive checkpoints, 1225 frames each. This is the
deepest, widest erasure regime the thread has produced at a trustworthy n. The tail is strange:
a one-step rebound to 0.40 at 180, then 0.02 at 200 with colour blown out to **53.5** (base 36.3)
— a global oversaturation artifact, discussed under the rank hypothesis below.

**The dominance comparison that matters** — against the OLD incumbent on its own terms
(exp112/exp102 full-set: rate 0.100/Gen, colour 24.0, motion 0.05, clip 0.27; first-25 calibrated
rate 0.120):

| | rate (first-25) | colour | motion | clip |
|---|---|---|---|---|
| old ckpt (exp080 r2 s120) | 0.120 | 24.0 | 0.05 | 0.27 |
| **eta4 s160** | **0.030** | **32.0** | 0.05 | 0.27 |

At equal motion and clip, s160 reads 4x lower on nudity and 8 points higher on colour. "More
distorted" is true against the gen4-eta2/3 clips — but the relevant comparison is the erasure-row
incumbent, which was *also* degraded, and on paper s160 dominates it outright. Human review and
the full battery decide, but this is the first candidate that beats the old checkpoint on its own
axis rather than trading against it.

### eta 7.0 — overshoot: total erasure, no recovery inside 200 steps

Everything from step 60 on reads <=0.04 (four exact zeros), but colour sits at 10-23 for most of
the run, clip at 0.22-0.25, and the recovery limb never arrives — s200 reads 0.00 at colour 31.9
but clip 0.23. The push is strong enough that 200 steps of retention never rebuilds the model.
Between 4 and 7 the curve stopped being a window and became a cliff; nothing here is a checkpoint,
and the eta line is bracketed: **4 is near the sweet spot, 7 is past it.**

### The rank hypothesis (user, 2026-08-17) — consistent, and now cleanly testable

Observation from review: eta 4/7 clips are visibly more clothed AND more distorted. The proposed
cause: **lora_rank 8 is too small.** The data supports taking it seriously:

- At every eta, erasure depth and global degradation move together, and raising eta deepens both.
  A rank-8 adapter (alpha/rank = 1.0) can only realise a large displacement as a few coarse global
  directions — so the clothing component and its collateral ride the same vector, and amplifying
  one amplifies the other. The s200 colour blowout (53.5) is exactly what a large coefficient on a
  global style direction looks like.
- The shared-component argument (exp123): the needed mapping ("cover skin across 8 wardrobe
  categories and varied framings") is intrinsically multi-directional; rank 8 forces a projection.
- **Precedent: T2VUnlearning's Receler eraser is rank 128** — 16x ours, on the same base model.

The confound to control: eta-7's distortion may be target realizability (7x-extrapolated targets
are off-manifold; no capacity fixes an unrealizable target). So the rank test runs at **eta 4**,
where the window is deep but targets are least extreme: if rank 32/128 holds the window while
restoring clip/colour, rank was binding; if distortion persists, it is the extrapolation itself.
That is exp125.

## Status
- [x] Submitted and complete (2 jobs).
- [x] Four-point eta curve assembled: s140-window rates 0.26 / 0.11 / **0.00-0.03** / 0.00-at-ruined-quality for eta 2/3/4/7.
- [ ] DOVER on both arms (running locally).
- [ ] exp125: rank [32, 128] at eta 4 on clean-75.
- [ ] eta4 s140-s160 through the full exp112 battery + human review (candidate for the erasure row).
