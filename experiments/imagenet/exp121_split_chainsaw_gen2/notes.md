---
status: done
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  12/30 (40%) against exp117's 14/30 (47%) — the seed control passes, so exp117's yield is a property
  of the prompts and not luck, and re-seeding a validated prompt set is a reliable ~1.6 h way to buy
  rows. Survivors skew 9 first / 3 second, which still leaves the merged chain-saw set balanced at
  33 rows / 19 first / 14 second (exp117 14 + exp066 7 + these 12). Zero blank targets. Dataset for
  exp123.
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

## Results (`outputs_20260816_001953`, helios, 1 h 37 m)

| | pass | not-split | no-concept | blank-target | first / second |
|---|---|---|---|---|---|
| exp117 (reference) | 14/30 (47%) | 4 | 12 | 0 | 6 / 8 |
| **exp121** | **12/30 (40%)** | 7 | 11 | 0 | 9 / 3 |

- **The seed control passes.** 40% against 47% is the same rate within the noise of a 30-row draw, so
  exp117's yield is a property of the prompt set, not of its seeds — and re-seeding a validated prompt
  set is a dependable way to buy rows at ~1.6 h per 30 triples. That is now the thread's cheapest
  scaling route, since exp120 showed the sampler knob does not raise the rate.
- `no-concept` did not climb (11 vs 12), as expected from identical prompts. `not-split` rose 4 → 7,
  which is where the missing two passes went — the substitute holding the concept is a per-seed
  accident, not a prompt property.
- **Region skew, and why it does not matter here.** Survivors are 9 first / 3 second, the opposite lean
  from exp117's 6/8. Merged, the chain-saw set is 33 rows at 19 first / 14 second — still balanced
  enough that the positional shortcut has nothing to lock onto.
- Zero blank targets under the new gate (see exp122 — church hit this and chain saw did not).

## Downstream
exp123 trains on the 33-row merge (exp117 + exp066 + this). exp069 trained on the 21-row set.

## Status
- [x] Submitted; completed 2026-08-16 (job 20735917).
- [x] Screened → `outputs_20260816_001953_screened.json`, 12 entries.
- [x] Pass count compared against exp117's 14/30.
- [ ] Merged with exp117 + exp066 for exp123 (`merge_dataset.sh`, run on the cluster).
