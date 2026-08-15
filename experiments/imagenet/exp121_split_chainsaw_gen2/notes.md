---
status: ready
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  Second generation of the chain-saw dataset — exp117's prompts verbatim under seeds 3231-3260. At
  exp117's measured 47% this adds ~14 rows, taking the merged chain-saw set to ~35. Also the seed
  control: exp116 showed re-seeding reproduces a prompt set's yield, so a result far from 14/30 would
  mean exp117's 47% was luck. Not submitted yet.
---
# exp121 — chain-saw dataset, generation 2

## Goal
More usable rows, by the boring route. exp117 took the prompts from 7/30 to 14/30; this run samples
the same prompt distribution again rather than trying to raise the rate further. exp120 is the
experiment that tries to raise the rate — the two are deliberately independent, so a failure there
does not leave the thread short of data.

## Setup
`prompts/imagenet_objects/split/chain_saw_closeup_gen2.csv` — the exp117 triples with seeds shifted
+30 (`tools/build_split_imagenet_closeup_prompts.py`, `GEN2_SEED_OFFSET`). Every sampler field is
exp117's, so the two builds merge into one homogeneous dataset with
`merge_dataset.sh`. `emit_whole_clip_target` is off: it answered its question in exp117 and the
variant is not a usable training target (see the config).

## What to watch
- **Pass count.** ~14/30 confirms the yield is a property of the prompts. Far below means exp117 was
  seed luck and the merged set is weaker than it looks; far above is equally worth knowing.
- **`no-concept` count.** exp117's was 12. It should not climb — these are the same prompts.
- **`concept_region` balance among survivors**, which decides whether the merged set stays free of
  the positional shortcut. exp117 contributed 6 first / 8 second and exp066 4 first / 3 second.

## Downstream
Merge with exp117's screened set (and exp066 run 2's) for the second chain-saw training run. exp069
trains on what exists now; this is what the follow-up trains on.

## Status
- [ ] Submitted.
- [ ] Screened (`tools/screen_split_dataset.py --min-concept-max 0.10 --write-filtered`).
- [ ] Pass count compared against exp117's 14/30.
- [ ] Merged with exp117 + exp066 for the follow-up run.
