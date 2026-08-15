---
status: ready
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  frame_replace erasure of 'church', in exp069's exact regime, isolating how much the method
  depends on the concept being localized. Unblocked by exp118; trains on its 14 screened rows, which
  carry a 10-first / 4-second positional skew — the concept eval set is the test that catches a
  shortcut LoRA, and exp122 rebalances the data.
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
Identical to exp069 apart from the dataset, `concept_target`, `retention_exclude` and the control
prompt files.

**Dataset: exp118's 14 screened rows**, no merge. exp067 run 2's 3 survivors are excluded on purpose:
all three are `concept_region: first`, and exp118's set is already 10 first / 4 second — three more
would take it to 13/4 for a 21% size gain. exp122 draws fresh seeds to rebalance instead. (This is
the opposite call from exp069, where the older build's rows both balanced the sides and added the
framing diversity church does not need — its clips are all wide by nature.)

**Read the skew, not around it.** A 10/4 keep set can be satisfied by the positional shortcut "copy
the concept-free half onto the other". The 20 church eval prompts are ordinary full scenes with no
object-free half, so they cannot be lowered by a shortcut LoRA — which makes the concept-set curve
the shortcut test, exactly as designed.

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
- [x] Datasets complete; config wired to exp118's screened set and exp068's anchors.
- [ ] Submitted.
- [ ] Checkpoint chosen for exp071; chain-saw vs. church comparison written up.
