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

## Conclusions

**Best run: run_005 (`ngs=1.0, lr=0.0005`)**

Achieves full fire erasure at step 400 while preserving the highest CLIP scores (~0.30 avg).
Step 800 is also notable: all three FDR rates reach 0.0 simultaneously (no spurious fire anywhere),
though CLIP drops slightly (avg 0.287).

Higher `ngs` (1.5, 2.0) and higher `lr` (0.001) achieve erasure faster but over-perturb the model,
resulting in significantly degraded CLIP scores — especially on related prompts.

## Suggested Next Steps

- Focus on `ngs=1.0`, refine `lr` in the 0.0003–0.0005 range.
- Run_005 should be extended to full 1000 steps with more eval prompts to confirm stability.
- Consider early stopping around step 400–600 based on concept_fdr reaching 0.
