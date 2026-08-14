---
status: active
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  frame_replace erasure of 'church', in exp069's exact regime, isolating how much the method
  depends on the concept being localized.
---
# exp070 — frame_replace erasure of "church"

## Goal
The hard half of the pilot, run in exactly exp069's regime so the two are directly comparable. A
church fills the frame and defines the scene; a chain saw sits inside one. Comparing the two isolates
how much frame_replace depends on the concept being *localized*, which is the property
`docs/comparison_targets.md` §2.2 claims makes objects its native regime.

Reference point: T2VUnlearning's per-class ESR-1 is 100 on garbage truck and French horn but 82.35 on
church — the class every method finds hardest.

## Setup
Identical to exp069 apart from the dataset (exp067), `concept_target`, `retention_exclude` and the
control prompt files. Replace the `outputs_TIMESTAMP` placeholders before submitting.

`./submit_job.py helios experiments/imagenet/exp070_frame_replace_church/config.yaml`

## What to watch
- Same three reads as exp069 (erasure, shortcut, collateral), plus:
- **What replaces the church.** If erasure works by substituting a specific building (the B-prompt
  substitutes were varied precisely to avoid this), that shows up as coherent buildings in the eval
  videos rather than absence. Worth eyeballing `eval_step_*/concept/`.
- **Scene damage.** Removing a frame-filling structure risks taking the surrounding scene with it —
  watch clip score and colorfulness on the unrelated set more closely than in exp069.

## Downstream
exp071.

## Status
- [ ] exp067 and exp068 complete; timestamps filled in.
- [ ] Submitted.
- [ ] Checkpoint chosen for exp071; chain-saw vs. church comparison written up.
