---
status: done
concept: nudity
method: eval
thread: nudity
takeaway: >
  Base-model row on T2VUnlearning's two utility dimensions (Object Class, Subject Consistency).
  DONE — 151 clips generated. Also establishes a result we needed independently: NudeNet's
  frame-level FALSE-POSITIVE FLOOR on the base model is 0.05 / 0.04 on prompt sets containing no
  nudity, so every nudity rate we and they report sits on a scale whose zero is ~4-5 points up.
  Subject Consistency and Object Class scoring is post-hoc and local, pending the video pull.
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

## Results (2026-08-10) — generation complete, both runs

| | prompts | clip | colorfulness | motion | nudity frame rate |
|---|---|---|---|---|---|
| run_001 `object_class` | 79 | 0.28 | 45.79 | 0.92 | **0.053** (206/3871) |
| run_002 `subject_consistency` | 72 | 0.31 | 40.46 | 1.60 | **0.043** (152/3528) |

### The by-product: NudeNet has a 4-5% false-positive floor

The detector was run over these clips only as a sanity check — a nudity eraser should not be
suppressing "a person" in general. It produced something more useful. On 79 bare COCO nouns and 72
"a person doing X" prompts, with no nudity anywhere in either set, the **base model** reads
**0.053** and **0.043** at frame level.

That is the floor of the instrument, not content. It matters for every number in
[`docs/comparability_t2vunlearning.md`](../../docs/comparability_t2vunlearning.md): the base model's
0.414 on the Gen set, and T2VUnlearning's 61.80, are both measured on a scale whose zero is around
0.04-0.05. It does not change any ranking — every method is measured on the same offset scale — but
it does mean a reported rate near 0.05 is at the noise floor and should not be described as residual
nudity. Our best checkpoint reads 0.0000 on the Gen concept set, i.e. *below* the base model's floor
on unrelated content, which is worth stating plainly rather than leaving for a reviewer to notice.

### Utility, against exp107

Full A/B is in [exp107](../exp107_vbench_utility_frame_replace/notes.md). Summary: CLIP is
unchanged, colorfulness is essentially preserved on general prompts (-8.5% on one set, **+8.2%** on
the other), and motion falls -68% / -36%. So the colour destruction we see on nudity prompts (-40%)
is the erasure doing its job, while the motion collapse is global.

## Status
- [x] Prompt sets built and committed.
- [x] Submitted and complete (2 jobs, 151 clips).
- [x] NudeNet floor measured (0.053 / 0.043).
- [ ] Subject Consistency scored — needs the videos pulled (`pull_results.sh` without `--no-videos`).
- [ ] Object Class instrument chosen and scored.
