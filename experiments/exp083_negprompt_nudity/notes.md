---
status: ready
concept: nudity
method: eval
thread: nudity
takeaway: >
  NegPrompt baseline for nudity — no training, concept passed as an inference-time negative prompt.
  Answers the first question any reviewer asks an erasure paper ("why not just negative-prompt
  it?"). Same two external benchmarks and seeds as exp082, so the pair is a clean A/B. Costs
  generation only. Not yet submitted.
---
# exp083 — NegPrompt baseline (nudity)

## Why
A reviewer's first objection to any erasure method is that a training-free negative prompt might do
the same job for free. T2VUnlearning compares against exactly this baseline, and the ImageNet side
of this project already runs it (exp065, `negative_prompt: auto`). Nudity had no such row, so
"frame_replace erases nudity" had nothing cheap to be measured against. This fills that row for one
generation job per benchmark and no training.

If NegPrompt turns out to match us, that is far better learned now than from a reviewer in
September.

## Setup
Identical to exp082 — same base model, same two external benchmarks, same `(prompt, seed)` pairs,
same unrelated control — with exactly one difference: `negative_prompt` is set. So the A/B isolates
the intervention and nothing else. Grids over `control_concept_prompts` into run_001 (I2P) and
run_002 (SafeSora).

**Negative prompt**: `"nudity, naked, nude, bare skin, explicit sexual content"`.

Two deliberate choices worth defending in the paper:

- **Several synonyms, not the bare word "nudity".** A one-word negative prompt would be a weak
  baseline that we beat by under-specifying it; a real deployment would list synonyms. The ImageNet
  side negates just the class name because there a class name *is* the complete concept, whereas
  "nudity" has common near-synonyms. If anything this biases *against* us, which is the right
  direction for a baseline.
- **Applied to every prompt set, including `unrelated`.** NegPrompt is a deployed inference-time
  defence — it is on for all generations, not selectively for concept prompts. The collateral damage
  it does to unrelated prompts is therefore part of what is being measured, exactly as PSR captures
  it on the object side. Implemented in `zml/unlearn/eval.py:evaluate`; recorded as
  `_negative_prompt` in the run's `metrics.json` so the output dir is self-describing.

## What to watch
- **The tradeoff, not just the erasure number.** NegPrompt's known failure mode is that it degrades
  generation broadly — that is the whole reason trained erasure exists. The interesting row is
  concept detection *together with* unrelated `clip_score_mean`/`motion`/`colorfulness`. "NegPrompt
  erases as well as we do but wrecks unrelated prompts" is a perfectly good result for us;
  "NegPrompt erases as well and costs nothing" is a problem we need to know about now.
- Whether it behaves differently on the two prompt distributions (I2P's long art prompts vs
  SafeSora's short blunt ones) — CFG-based steering is sensitive to how much of the prompt is
  about the concept.
- Per [[feedback-detector-metrics-not-ground-truth]], a low NudeNet rate here needs the same visual
  spot-check we give our own runs before it is believed; do not hand NegPrompt a pass on the
  detector alone that we would not accept for frame_replace.

## Status
- [x] `negative_prompt` support added to the shared eval path (`eval_model.Config` +
      `zml/unlearn/eval.py`); verified it is a strict no-op when unset, so no prior run changes.
- [x] Config drafted; grid + `Config` construction verified end-to-end locally.
- [ ] Submitted — manual, per project convention.
- [ ] Compared against exp082 (base) on the same pairs; tradeoff column read, not just erasure.
- [ ] Visual spot-check before believing any low detection rate.
