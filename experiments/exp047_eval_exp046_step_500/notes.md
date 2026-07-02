# exp047 — verify the exp046 step-500 checkpoint on full prompt sets

## Hypothesis
exp046 (frame_replace redirection) produced the project's best erasure at step 500:
concept `fire_detection_rate` 0.0 and `fire_area` ~0.0002 with unrelated clip_score held at
0.33. But the live eval uses only 5 concept prompts (±0.2 detection granularity), the
checkpoint-to-checkpoint trajectory oscillates (1.0 at 400 → 0.0 at 500 → 0.6 at 600), and
2 of the 5 step-500 concept videos are near-grayscale (colorfulness 0.02 / 2.77, clip
0.23 / 0.27) — so part of the "win" may be eval luck plus quality collapse rather than clean
fire removal. Before building on this checkpoint (exp048 stabilized rerun), verify it on the
full 15-prompt fire set.

Also introduces `prompts/cogvideox_fire_control_related_v2.csv`, a *hard* related set:
clearly fire-free scenes (no lava, candles, fireworks, embers) that share fire's colors and
motion (swirling autumn leaves, rippling orange silk, koi, marigold petals, monarch swarm,
dust devil, cold-inflated balloon envelope, storm-tossed red maples, confetti updraft, silk
ribbons). The v1 related set was ambiguous — lava/candle/molten-glass prompts border on fire,
so its signal never separated erasure damage from legitimate concept removal. Seeds are baked
per prompt (seed policy) and committed once.

## Pipeline
`./submit_job.py athena experiments/exp047_eval_exp046_step_500/config.yaml`
(checkpoint weights live on the cluster in the exp046 outputs dir; nothing to precompute).

## What to watch
- **Concept (15 fire prompts):** `fire_detection_rate` staying near 0 with `clip_score_mean`
  ≥ ~0.30 confirms the win is real. Detection bouncing back toward 0.6–1.0 means step 500 was
  5-prompt luck and the stabilization rerun (exp048) carries the track.
- **Grayscale collapse check:** concept `colorfulness_mean` and per-video scores — the live
  eval showed 2/5 videos at colorfulness ≈ 0. If a similar fraction of the 15 videos is washed
  out, the erasure is partly quality destruction and exp049 (quality-preserving variant) moves up.
- **Related v2 (10 hard prompts):** should be untouched — detection ~0 *and* clip/colorfulness
  at healthy levels. Degradation here means the erasure bleeds into fire-colored/fire-motion
  content, which the v1 set could never show cleanly.
- **Unrelated (15 prompts):** clip ~0.33, colorfulness not cratering (same bar as exp046).

## Results
- (pending run)
