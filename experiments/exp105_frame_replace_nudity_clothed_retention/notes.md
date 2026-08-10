---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  Isolates the RETENTION variable: exp104's fully-clothed anchors on exp080's OLD dataset, eta
  [1.5, 2.0]. Paired with exp088, which isolates the data variable on the same eta pair. Blocked on
  exp104 (two retention paths are placeholders). 2 jobs.
---
# exp105 — clothed retention anchors, old data

## Why
exp085 erased *worse* with nudity-specific anchors than exp086 did with fire-era ones. The cause is
compositional and is written up in [exp104's notes](../exp104_clothed_retention_precompute/notes.md):
exp079's human-filtered set is **11/20 exposed-skin wardrobe**, so retention was pulling toward
keeping exposed torsos while the erase term pushed away from the same features. exp041's fire
anchors never compete, which is why the "wrong" set won.

exp104 rebuilds the anchors fully clothed while keeping the **shot grammar** of the training targets
— disjoint in content, overlapping in the regions the erase term damages. That is the difference
from exp041, and the reason to expect it to beat both.

## The hypothesis this actually tests
Erasure is currently coupled to degeneracy. In exp086 run_003 the frame rate reads **0.01 at step 70**
(colorfulness 18.6), **0.21 at step 120** (23.4), **0.76 at step 170** (33.4) — strongest erasure
exactly where the model is most degraded, decaying monotonically as it recovers. Every arm of
exp085/exp086 shows the same U-shape, and human review of exp080 described it directly (nude →
distorted → clothed → nude again).

Anchors that pin "human, clothed, moving, colourful" should let erasure **survive the recovery
limb** instead of washing out with it. If it works, the frame rate stays low while colorfulness
climbs back toward base — which is a different curve shape, not just a better number, and that is
what to look for.

## One variable
| | data | retention | eta |
|---|---|---|---|
| exp080 run_002 *(have)* | old | fire | 2.0 |
| exp086 run_003 *(have)* | old | fire | 1.5 |
| exp088 | **clean (exp087)** | fire | 1.5, 2.0 |
| **exp105 (this)** | old | **clothed (exp104)** | 1.5, 2.0 |

Deliberately **not** on exp087's clean data: exp088 is testing that separately, and changing both at
once would make a mediocre result uninterpretable. If both arms help, a combined run follows.

## Blocked on
`retention_metadata_file` / `retention_latents_dir` are placeholders — exp104's output directory does
not exist until that job runs. Fill both in after exp104 is reviewed; `slurm/check_config_paths.sh`
will refuse the submission otherwise, which is the intended safety net.

The filtered metadata must sit at exp104's **experiment root**, not under `outputs_*/` — that path is
gitignored and would never reach the cluster (the trap that aborted exp085's first submission).

## What to watch
- **Curve shape, not just the minimum.** Does low frame rate persist as colorfulness recovers?
- **Motion against exp086 run_003 at matched step.** exp086's eta 1.5 bottoms out around 0.10-0.23
  against a base of 0.686. Anything above that at equal erasure is the win condition.
- **Read `nudity_frame_rate`, not `nudity_detection_rate`.** `eval_num_prompts: 10` means the video
  rate cannot resolve 0 vs 1 clip; the frame rate is over 490 frames. Both are emitted now.
- Per [[feedback-detector-metrics-not-ground-truth]], the winning checkpoint needs human review
  before any number is reported — exp080's phase structure was invisible to every metric.

## Status
- [ ] exp104 run and reviewed; retention paths filled in.
- [ ] Submitted (2 jobs).
- [ ] Trajectories overlaid against exp086 run_003 and exp080 run_002.
- [ ] Human review of the best checkpoint.
