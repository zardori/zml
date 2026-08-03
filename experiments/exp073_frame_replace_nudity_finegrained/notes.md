---
status: active
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  Zooming into exp062's 0-100 step window (eval every 20 steps) to separate genuine erasure from
  the generation-collapse artifact found at exp062's step-100 checkpoint.
---
# exp073 — frame_replace nudity, fine-grained early checkpoints (0-100 steps)

## Goal
exp062 (run 2, human-filtered 12-triple dataset) showed `nudity_detection_rate` at its lowest
(0.1-ish across both runs) at the very first checkpoint (step 100), then climbing back up through
step 600. Cross-checking video file sizes and `motion_score_mean` per checkpoint showed the step-100
checkpoint was mostly **generation collapse** (8/10 concept videos near-blank, motion_score_mean
~0.03 vs. ~0.34 by step 600 once generation stabilized) — the same blank/degenerate-frame behavior
seen in exp063's base-model baseline, not necessarily genuine clothed erasure. The "nudity reappears"
trend may just be collapse resolving, not erasure fading.

This run adds checkpoints every 20 steps (20/40/60/80/100) to see:
1. Whether there's a step in 0-100 where the model is both **non-collapsed** (motion_score_mean back
   to a normal range, videos not blank) **and** shows a genuinely low `nudity_detection_rate`.
2. Or whether collapse and low detection are inseparable at this LR (5e-4, constant) — i.e. any point
   where generation is coherent already shows nudity back near baseline.

## Setup
Identical to exp062 run 3 (21-triple human-reviewed dataset, retention, erase regime, seed) — only
`steps: 100` and `save_interval: 20` differ, so this is a pure zoom-in, not a new variable. (Originally
pointed at run 2's 12-triple dataset; repointed to run 3's 21-triple set once that landed, since
exp073 hadn't been submitted yet.)

## What to watch
- Per-checkpoint video file sizes / `motion_score_mean` (collapse signal) alongside
  `nudity_detection_rate`, the same way exp062 was re-analyzed. A blank/degenerate video should not
  be read as "erased."
- If no clean non-collapsed low-detection point exists in 0-100, that's evidence the LR is too high
  for this dataset size (12 triples) — collapse and erasure are two faces of the same
  too-large-a-step early on, worth a lower-LR or warmup follow-up rather than more checkpoint
  granularity.

## Status
- [ ] Submitted.
- [ ] Analysis.
