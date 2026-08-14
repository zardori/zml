---
status: done
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  SPLIT RESULT, and it rewrites the limitation section. Clothed anchors PROTECT MOTION — 2-3x the
  fire-retention runs across the whole back half (0.35 vs 0.13 at step 130), a curve-level effect,
  not noise. So the motion collapse is NOT intrinsic to eta-extrapolated erasure, which is what we
  were about to write. But they also BLOCK ERASURE: the rate never reaches a stable zero (best are
  isolated dips, 0.02 @ r1 s50, 0.04 @ r2 s60, both contradicted by their neighbours). Same failure
  mode as exp085, weaker. Retention weight is the untried knob — this ran at 1.0.
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

## Results (2026-08-11) — both runs complete, 200 steps

### The win condition was met: clothed anchors protect motion

Concept motion against base **0.686**, at matched steps, eta 1.5 throughout:

| step | 50 | 70 | 90 | 110 | **130** | 150 | 170 | 190 |
|---|---|---|---|---|---|---|---|---|
| **exp105 r1** (clothed) | 0.34 | 0.19 | 0.20 | **0.24** | **0.35** | **0.28** | **0.19** | **0.21** |
| exp086 r3 (fire) | 0.35 | 0.23 | 0.18 | 0.17 | 0.13 | 0.10 | 0.11 | 0.11 |
| exp088 r1 (fire, clean data) | 0.29 | 0.19 | 0.16 | 0.14 | 0.07 | 0.10 | 0.08 | 0.09 |

The three curves are indistinguishable through step ~90 and then separate cleanly. From step 110 on,
clothed retention holds **2–3x** the motion of either fire-retention run, and it *recovers* (0.20 →
0.35) where the fire runs keep sliding (0.18 → 0.13, 0.16 → 0.07). This is a curve-level effect over
ten checkpoints, not the isolated-zero noise that has fooled us twice in this thread.

**So the motion collapse is not intrinsic to eta-extrapolated v-prediction erasure.** That is the
sentence [exp088](../exp088_frame_replace_nudity_clean/notes.md) left us about to write as a
limitation, and it is wrong: retention *can* hold motion, provided the anchors contain moving people.
exp041's fire anchors have nothing human to pin, which is why they never protected it.

### But erasure is blocked

| | best rate | at step | motion | colour |
|---|---|---|---|---|
| exp105 r1 (clothed, eta 1.5) | 0.0200 | 50 | 0.34 | 22.7 |
| exp105 r2 (clothed, eta 2.0) | 0.0400 | 60 | 0.21 | 19.3 |
| exp086 r3 (fire, eta 1.5) | 0.0100 | 70 | 0.23 | 18.6 |
| exp080 r2 (fire, eta 2.0) | **0.0000** | 120 | 0.11 | 21.9 |

Neither exp105 arm reaches a stable zero, and both "best" points are **isolated single-step dips**:
r1's 0.02 at step 50 sits between 0.23 (s40) and 0.19 (s60); r2's 0.04 at step 60 sits between 0.20
and 0.15. Neighbouring checkpoints contradict them, so neither is a regime. The rest of both
trajectories runs at 0.2–0.47 — worse erasure than fire retention at every comparable point.

This is exp085's failure mode again, weaker. Even *fully clothed* people anchors compete with the
erase term, because what they share with the concept is not wardrobe but human-body features. The
[design rule from exp104](../exp104_clothed_retention_precompute/notes.md) — retention must be
semantically disjoint from the concept — turns out to bind harder than "no exposed skin".

### The frontier moved, though

At **matched erasure** the clothed arm is genuinely Pareto-better: at rate 0.02, exp105 r1 gives
motion 0.34 / colour 22.7 against exp086 r3's 0.16 / 16.8 — 2.1x the motion and +35% colour. The
caveat above applies (r1's 0.02 is a transient), so this is a direction rather than a result.

### The untried knob

Both arms ran at `retention_weight: 1.0`. The two ends are now measured — fire anchors erase but kill
motion, clothed anchors keep motion but do not erase — and nothing between them has been tried. A
weight sweep on the clothed set (0.25 / 0.5) is the obvious next experiment and is cheap: same data,
same eta, one field. That is a much better use of GPU time than another dataset or eta variation,
both of which are now settled as nulls.

## Status
- [x] exp104 run and reviewed; retention paths filled in.
- [x] Submitted and complete (2 jobs, 200 steps each).
- [x] Trajectories overlaid against exp086 run_003 and exp080 run_002.
- [ ] DOVER scored locally (helios wrote 0.0) — needs the eval videos pulled.
- [ ] Human review — particularly of the step 110–150 window, where the motion claim lives.
- [ ] `retention_weight` sweep on the clothed anchors.
