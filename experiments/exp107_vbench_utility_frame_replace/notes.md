---
status: ready
concept: nudity
method: eval
thread: nudity
takeaway: >
  METHOD row on T2VUnlearning's two utility metrics, pairing with exp106 (base). Expect Subject
  Consistency to score our ~-88%-motion checkpoint AT OR ABOVE base — the metric reading a defect
  as a strength, which is the demonstration that their instruments miss temporal collapse. Object
  Class is where we may legitimately lose. 2 jobs. Checkpoint is exp080 run_002 step 120.
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

## Status
- [ ] Submitted (2 jobs, after or alongside exp106).
- [ ] Subject Consistency scored against exp106's base.
- [ ] Object Class scored against exp106's base.
- [ ] Checkpoint repointed if exp088/exp105 win.
