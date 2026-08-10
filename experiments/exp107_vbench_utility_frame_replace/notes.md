---
status: done
concept: nudity
method: eval
thread: nudity
takeaway: >
  METHOD row on T2VUnlearning's two utility dimensions. DONE. Result that matters: on prompts with
  no nudity in them, CLIP is unchanged (0.28/0.30 vs 0.28/0.31) and colorfulness is preserved
  (-8.5%, +8.2%) — so the -40% colour loss on nudity prompts is the ERASURE, not general damage.
  Motion falls -68% / -36%, which locates the motion collapse as a global property of the adapter
  and independently refutes exp088's frozen-donor diagnosis. Subject Consistency / Object Class
  scoring pending the video pull.
---
# exp107 — frame_replace on VBench Object Class + Subject Consistency

## Why
Identical to [exp106](../exp106_vbench_utility_base/notes.md) except the checkpoint, so the pair is
a clean A/B. Rationale for reporting metrics we distrust is there and in
[`docs/comparability_t2vunlearning.md`](../../docs/comparability_t2vunlearning.md); the short version
is that publishing only our own instruments — DOVER, motion, CLIP, colorfulness, all of which make
our method look worse — invites the objection that theirs were left out deliberately.

## What to watch, and what to write down before seeing it
**Subject Consistency should be reported even if it flatters us, and especially then.** It measures
similarity to the first frame and to the previous frame, so a frozen clip approaches 1.0. This
checkpoint costs ~-88% motion. If it scores at or above base, that is not a preservation result —
it is evidence that the metric cannot see the failure. Writing that expectation down *now* is what
separates a finding from a post-hoc rationalisation.

> **Prediction confirmed on proxy clips, before this run (2026-08-10).** Scoring the same checkpoint's
> existing I2P clips: base **94.23** → ours **99.21**, a **+4.98** gain, against motion −88% and
> DOVER technical −18% on those same clips. T2VUnlearning's own method *loses* 0.83 on this metric
> (95.53 → 94.70), so on this instrument alone we would appear to preserve capability better than
> they do while having frozen the video. These are I2P nudity prompts rather than VBench's 72
> `subject_consistency` prompts, so this run is still needed for a number comparable with 94.70 —
> but at +5 and near the 100 ceiling, the effect will almost certainly reproduce.

**Object Class is the honest counterweight.** It asks whether the named object is actually
generated, and our concept-prompt CLIP score falls 0.30 → 0.23. If we lose here, we report it.

**The nudity detector runs over these clips too**, as a free check: a nudity eraser should not be
suppressing "a person" in general. A non-trivial nudity rate on VBench's `object_class` prompts
would also be worth knowing, since "a person" is exactly the prompt where over-erasure would show.

## Checkpoint
`exp080 run_002 step 120` — the best point human review found, still unbeaten after exp085 and
exp086. If exp088 or exp105 produce a better one, repoint `lora_checkpoint_dir` and change nothing
else; the prompts and seeds are fixed, so the comparison survives the swap.

## Results (2026-08-10) — generation complete, both runs

Against [exp106](../exp106_vbench_utility_base/notes.md), same prompts and seeds, checkpoint
`exp080 run_002 step 120`:

| set | metric | base | ours | delta |
|---|---|---|---|---|
| `object_class` (79) | clip | 0.28 | 0.28 | 0.0 |
| | colorfulness | 45.79 | 41.92 | **-8.5%** |
| | motion | 0.92 | 0.29 | **-68%** |
| | nudity frame rate | 0.053 | 0.010 | -81% |
| `subject_consistency` (72) | clip | 0.31 | 0.30 | -0.01 |
| | colorfulness | 40.46 | 43.76 | **+8.2%** |
| | motion | 1.60 | 1.03 | **-36%** |
| | nudity frame rate | 0.043 | 0.010 | -77% |

### The colour loss is the erasure; the motion loss is collateral

This is the result worth carrying into the paper. On nudity prompts this checkpoint costs **-40%**
colorfulness, which has read as general degradation for the whole thread. On prompts with no nudity
in them, colorfulness is *not* damaged: -8.5% on one set and **+8.2%** on the other. Two sets
straddling zero is a preserved quantity. CLIP is flat on both (0.28, 0.30).

So the -40% is concentrated on exactly the prompts the method is supposed to change — it is the
erasure operating, not the model getting worse. That converts what looked like our worst utility
number into evidence of localisation, and it is only visible because the general-prompt row exists.

**Motion does not behave that way.** -68% and -36% on nudity-free prompts is real collateral damage,
consistent with the -84..-88% on concept prompts. The collapse is a global property of the adapter.
This independently refutes the frozen-donor diagnosis that [exp088](../exp088_frame_replace_nudity_clean/notes.md)
was built to test and separately disproved: no change to the *nudity* training targets could fix
damage that shows up on "a bicycle" and "a person doing X".

### Subject Consistency — prediction confirmed on the real prompt set

Scored locally with `tools/score_subject_consistency.py` (DINO ViT-B/16, VBench's formula), on
VBench's own 72 `subject_consistency` prompts:

| | Subject Consistency ↑ | motion |
|---|---|---|
| base (exp106 r2) | 94.24 | 1.60 |
| **ours** | **96.41 (+2.17)** | **1.03 (−36%)** |
| *T2VUnlearning, Hunyuan* | *95.53 → 94.70 (−0.83)* | *not reported* |

**We gain on their preservation metric on the very clips where we lose a third of the motion, while
their own method takes a penalty on it.** The expectation was written into this file before the run,
so it is a confirmed prediction rather than a rationalisation.

Smaller than the I2P proxy predicted (+2.17 vs +4.98), and the reason is instructive: these clips
still move (motion 1.03, not 0.09), and the metric's reward scales with stillness. On the near-frozen
I2P clips it paid +5; here it pays +2.2. That relationship *is* the finding.

### On the detector floor
Base reads 0.053 / 0.043 on sets containing no nudity — NudeNet's false-positive floor. Ours reads
0.010 on both, i.e. the eraser suppresses the detector's own noise as well. Recorded in exp106; it
sets the scale on which every reported nudity rate, ours and theirs, actually sits.

## Status
- [x] Submitted and complete (2 jobs).
- [x] CLIP / colorfulness / motion / nudity-rate A/B against exp106.
- [x] Subject Consistency scored against exp106's base: **96.41 vs 94.24 (+2.17)** at motion −36%.
- [ ] DOVER backfilled locally (running; helios wrote 0.0).
- [ ] Object Class scored against exp106's base (instrument still to be chosen: GRiT vs OWL-ViT).
- [ ] Checkpoint repointed if exp088 run_002 or exp105 win.
