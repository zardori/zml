---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  Retention-set ablation against exp080: identical run with exp079's 20 human-reviewed
  nudity-adjacent anchors instead of exp041's fire-era near-misses. Answers whether a
  concept-matched preservation set actually buys anything, which no nudity run has ever tested.
  Not yet submitted — narrow the LR grid to exp080's winner first.
---
# exp085 — retention-set ablation (nudity)

## Why
Every nudity `frame_replace` run to date — exp062, exp073, exp077, and exp080 as submitted — anchors
its retention branch on `exp041`, a set of 26 prompts written to be *fire* near-misses: monarch
butterflies, autumn maples, koi ponds, foxes, poppy fields, copper stills. exp041's notes call it
"concept-agnostic," but it isn't; every prompt was chosen to sit just outside fire. Used for nudity
at `retention_weight: 1.0`, it anchors preserved behaviour on the model's ability to render orange
animals, which is not the collateral surface a nudity eraser threatens.

exp079 built the matched replacement (30 nudity-adjacent clothed scenes: swimwear, athletic,
medical, sleepwear, bathing, parenting, clothed intimacy, close crops, multi-person) and human
review kept 20. Nothing has yet trained against it, so "a concept-matched retention set helps" is
currently an assumption, not a result.

This run tests it. exp080 is the control and this is the treatment; they share the merged 34-triple
dataset, the LR regime, and everything else.

## Setup
Byte-identical to exp080's submitted config except:

- `retention_metadata_file` / `retention_latents_dir` -> exp079's `metadata_human_filtered.json`
  (20 anchors) instead of exp041's `metadata.json` (25).
- `save_interval` 20 -> 40 and `eval_num_prompts` 10 -> 20. Eval runs at every `save_interval` over
  three prompt sets, so clips per run = `(steps / save_interval) * 3 * eval_num_prompts`. Both
  configurations come to 300 clips; this one spends the budget on statistical power per point (n=20
  rather than n=10) instead of on temporal resolution (5 checkpoints rather than 10). At n=10 a
  single clip moves the rate 10 percentage points, and exp073's whole trajectory — 0.0, 0.1, 0.1,
  0.1, 0.3 — is consistent with pure noise.
- `slurm_time` 16h -> 20h. exp080's 16h leaves no margin at 300 clips plus 200 steps, and exp082 and
  exp083 were both lost to an eval-phase timeout.

**Before submitting: narrow `learning_rate` to whichever value exp080 picks.** LR is not the
variable under test here, and running the full four-point grid again would spend 4x the compute to
answer a one-dimensional question. The grid is carried over only so the file is runnable as-is.

## What to watch
- **The `unrelated` column, not the concept column.** If the matched anchors work, erasure should be
  roughly unchanged while collateral damage drops. Equal erasure with better preservation is the
  win; better erasure would be surprising and worth understanding rather than celebrating.
- **Whether the retention anchors fight the erase objective.** exp079 found NudeNet scores its own
  anchors as nudity — 0.844 on a red bikini across all 49 frames — so on this content the two loss
  terms disagree by construction. A model that correctly preserves swimwear can be scored as "still
  generating nudity." This is the strongest argument for reporting a `related` column rather than
  the concept column alone, and it applies to reading this run's numbers.
- Per [[feedback-detector-metrics-not-ground-truth]], neither outcome is believable from the
  detector alone.

## Status
- [x] exp079's anchors built and human-reviewed (20/30 kept).
- [x] Config prepared; reuses exp080's merged `combined_dataset/`, so no new precompute.
- [ ] `learning_rate` narrowed to exp080's chosen value.
- [ ] Submitted (after exp080's grid completes, so the baseline is whole).
- [ ] Compared against exp080 on the `unrelated` and `related` columns.
