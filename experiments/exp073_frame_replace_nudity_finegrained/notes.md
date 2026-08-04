---
status: done
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  Non-collapse confirmed by both metrics and human review: no blank/frozen frames, and people in
  the clips genuinely have clothes (real erasure, matches exp062 run 3). But human review
  (2026-08-04) also found the clips visibly soft/not sharp — a distortion our current metrics
  (clip_score, colorfulness, motion) don't capture and DOVER can't either (unavailable in this
  environment, its 0.0 fields are a placeholder, not a real quality score). Colorfulness is
  lowest exactly at the earliest, least-sharp-looking checkpoints (30.9 @ step 20 -> 47.4 @ step
  100), consistent with — not proof of — early-training instability under the constant 5e-4 LR
  with no warmup. Worth trying LR warmup or a lower peak LR before assuming this softness is
  inherent to the method.
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

## Results (`outputs_20260803_123355`)

| step | concept det | motion | clip_score | colorfulness |
|--:|--:|--:|--:|--:|
| 20 | 0.0 | 0.398 | 0.287 | 30.9 |
| 40 | 0.1 | 0.250 | 0.295 | 38.8 |
| 60 | 0.1 | 0.095 | 0.291 | 38.2 |
| 80 | 0.3 | 0.223 | 0.283 | 42.3 |
| 100 | 0.1 | 0.199 | 0.285 | 47.4 |

No collapse signature (no near-zero motion + near-zero clip_score together) at any checkpoint —
the closest is step 60's low motion (0.095), but its clip_score (0.291) is normal, so that reads as
a static/low-motion generation, not a blank one. colorfulness climbs steadily 31→47 across the
window but stays well above the ~0 a blank frame would show.

**Answer to the original question: collapse and low detection are separable at this LR** — step 20
alone shows detection can already be 0.0 while generation is coherent, on this (larger, re-reviewed)
21-triple dataset. The collapse exp062 run 2 hit at step 100 was specific to the smaller 12-triple
dataset, not an inherent property of this LR/regime. Matches exp062 run 3's coarser-grained finding
(0.0 detection at steps 400/500, non-collapsed) — together these are the strongest signal yet that
frame_replace is transferring to nudity for real.

## Human video review (2026-08-04)

Confirms the metrics' "not collapsed" read qualitatively — people in these clips genuinely have
clothes, this is real erasure, not blank/degenerate frames. But the clips are **visibly
distorted / not sharp**, a defect none of our current per-checkpoint scores would catch
(`clip_score`/`colorfulness`/`motion` are semantic/color/movement proxies, not sharpness; DOVER —
the one metric in this pipeline actually meant to score technical quality — reads `0.0` at every
checkpoint here because it's simply not installed/available in this run's environment, not because
quality is truly zero. Don't read those DOVER fields as data until that's fixed.)

**Is this an LR/schedule artifact worth tuning?** Circumstantial evidence says maybe: colorfulness
is lowest at the earliest checkpoints (30.9 @ step 20, 38.8 @ step 40) and climbs steadily to 47.4
by step 100 — the same window where a `lr_scheduler: constant` schedule with no warmup hits full
5e-4 LR from step 1 on a rank-8 LoRA fine-tuned on only 21 triples. That's a plausible mechanism for
early softness that eases as training continues, distinct from the harder exp062-run-2 collapse
(which was near-blank, not just soft). Not proven — colorfulness is a color-vividness proxy, not a
sharpness measurement, so this is a correlation worth testing, not a diagnosis.

**Recommended next step, cheap to try:** rerun this same 0-100-step zoom with (a) a short linear LR
warmup (e.g. 20-50 steps) before the constant plateau, and/or (b) a lower peak LR (e.g. 2e-4-3e-4)
— compare colorfulness/clip_score at matched steps against this run, and eyeball the clips. If
sharpness improves without losing the erasure signal (detection still low), that's the fix; if not,
the softness is more likely inherent to this LoRA rank / dataset size than to the schedule, and a
higher-rank LoRA or more training data would be the next lever instead.

## Status
- [x] Submitted.
- [x] Analysis (above).
- [ ] LR-warmup / lower-peak-LR follow-up run, to test whether it's the fix for the softness.
