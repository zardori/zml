# exp005_esd_fire_grid — Grid Search Notes

## Setup

Grid search over `negative_guidance_scale` × `learning_rate`:
- `negative_guidance_scale`: [0.5, 1.0, 1.5, 2.0]
- `learning_rate`: [0.0002, 0.0005, 0.001]
- 12 runs total, up to 1000 steps, checkpoints every 200 steps

## Metrics

- `concept_fdr` — fire detection rate on fire-concept prompts (want **low**)
- `related_fdr` — fire detection rate on fire-related prompts (want low)
- `unrelated_fdr` — fire detection rate on unrelated prompts (want low, checks for spurious generation)
- `related_clip` / `unrelated_clip` — CLIP scores on related/unrelated prompts (want **high**, measures preserved model quality)

## Key Findings

### Runs that never achieved full erasure
- `lr=0.0002` (run_001, 004, 007, 010): concept_fdr stayed at 1.0 throughout — learning rate too low.
- `ngs=0.5` (run_001, 002, 003): insufficient negative guidance; best was concept_fdr=0.60 at step 1000.

### Runs achieving full erasure — ranked by CLIP preservation (avg of related+unrelated)

| Run    | ngs | lr     | Best step | c_fdr | r_fdr | u_fdr | avg_clip |
|--------|-----|--------|-----------|-------|-------|-------|----------|
| run_005 | 1.0 | 0.0005 | 400       | 0.00  | 0.20  | 0.00  | **0.301** |
| run_006 | 1.0 | 0.001  | 600       | 0.00  | 0.00  | 0.00  | 0.265    |
| run_008 | 1.5 | 0.0005 | 1000      | 0.00  | 0.00  | 0.00  | 0.252    |
| run_009 | 1.5 | 0.001  | 400       | 0.00  | 0.00  | 0.00  | 0.239    |
| run_011 | 2.0 | 0.0005 | 600       | 0.00  | 0.00  | 0.00  | 0.236    |
| run_012 | 2.0 | 0.001  | 200–400   | 0.00  | 0.00  | 0.00  | 0.183    |

## ⚠️ Correction — FDR overstates erasure here

The `concept_fdr → 0.0` readings in this grid are **misleading**. Inspecting the generated videos
shows FDR dropped mainly because the model **collapsed toward desaturated / black-and-white,
lower-quality** output — not because fire content was cleanly removed. Residual contours and
structure survived, so CLIP stayed ~0.30: CLIP is largely insensitive to this desaturation, and the
fire detector reads grayscale frames as "no fire."

So run_005 (`ngs=1.0, lr=5e-4`) is **not** a clean erasure — "FDR=0 + CLIP~0.30" conflates erasure
with quality collapse. Higher `ngs`/`lr` collapse harder. **Lesson: FDR alone is gameable by
desaturation**; evaluation needs a color/quality guard (a colorfulness metric is added in exp027).
The genuinely promising result is **exp006**, not any run here.

## Conclusions (revised)

The "full erasure" table above ranks by raw FDR/CLIP, which — per the correction — overstates
erasure. Treat those runs as **quality-collapse** runs, not clean removals. run_005 (`ngs=1.0,
lr=5e-4`) had the highest CLIP (~0.30) but reached FDR=0 by desaturating; at step 800 all three FDR
rates hit 0.0 (CLIP ~0.287). Higher `ngs` (1.5, 2.0) and higher `lr` (0.001) reach FDR=0 faster but
over-perturb the model and degrade CLIP further — consistent with the collapse mechanism above.

## Suggested Next Steps

- Do **not** chase FDR=0 here; it is reached via collapse. Use the colorfulness/quality guard
  (added in exp027) to distinguish genuine erasure from desaturation.
- The promising direction is exp006's larger, more diverse prompt set (better generalization, far
  less collapse) — build on that, not on these collapse runs.
