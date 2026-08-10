---
status: ready
concept: nudity
method: eval
thread: nudity
takeaway: >
  Base-model row on the two VBench dimensions T2VUnlearning use as their ENTIRE preservation
  evidence (Object Class, Subject Consistency). We report DOVER/motion/CLIP instead — instruments
  on which we look worse — so publishing only ours invites "you omitted theirs on purpose". Pairs
  with exp107. 2 jobs, 151 clips. Not yet submitted.
---
# exp106 — base model on VBench Object Class + Subject Consistency

## Why
T2VUnlearning's claim that their method "preserves the model's generation capability" rests on two
VBench dimensions and nothing else, reported for HunyuanVideo only:

| | Object Class ↑ | Subject Consistency ↑ |
|---|---|---|
| Original | 88.57 | 95.53 |
| SAFREE | 48.48 | 94.92 |
| T2VUnlearning | 87.00 | 94.70 |

For CogVideoX — the model we share — they publish **no** utility number at all. So there is no
like-for-like comparison to be had, and we cannot claim parity. What we can do is stop the omission
objection: report their metrics next to ours, and let the difference between the two make the
argument.

**The argument being: Subject Consistency rewards stillness.** It scores frame-to-first and
frame-to-previous feature similarity, so a frozen clip approaches 1.0. Our checkpoint costs ~-88%
motion and may therefore score *at or above* base on it. That is a demonstration that their
instruments cannot see temporal collapse — but only if the number is published, not asserted.

## Setup
VBench's own prompts, taken from the copy redistributed in **their** repo
(`evaluation/vbench_prompts/VBench_full_info.json`) so prompt drift between VBench releases cannot
silently break the comparison. Built by `tools/build_vbench_prompts.py`:
`prompts/vbench_object_class.csv` (79 bare COCO nouns) and
`prompts/vbench_subject_consistency.csv` (72 "a person doing X").

Both are *general-capability* probes on their own prompt sets, so neither can be scored on nudity
clips we already have — hence a generation run. Filed under `control_unrelated_prompts`: they are
unrelated to nudity, and the `concept` slot would file utility numbers under a key that reads as an
erasure rate.

**Deviation to state in the paper:** VBench samples 5 videos per prompt; we generate 1. Seeds are
hash-derived (VBench ships none) and frozen by committing the CSVs.

## Scoring — local, post-hoc, no cluster
- **Subject Consistency**: `zml/eval/subject_consistency.py`, DINO ViT-B/16 with VBench's exact
  formula. Sanity check on unrelated base clips reads 94.23, against their Hunyuan Original of
  95.53 — different model and prompts, same ballpark.
- **Object Class**: VBench/GRiT if detectron2 builds locally, else an OWL-ViT substitute with the
  deviation documented (same caveat we already carry for the nudity rate). Either way it touches
  only this machine, never the cluster.

## Status
- [x] Prompt sets built and committed.
- [ ] Submitted (2 jobs).
- [ ] Subject Consistency scored.
- [ ] Object Class instrument chosen and scored.
