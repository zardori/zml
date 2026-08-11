---
status: ready
concept: nudity
method: precompute
thread: nudity
takeaway: >
  Fourth-generation nudity training set, rebuilt because human review found gen1-gen3's edited
  clothing implausibly baggy or skin-toned — traced to B prompts written to satisfy "no bare skin
  visible", which selects for shapeless coverage. 200 triples (8 wardrobe categories x 25, seeds
  3801-4000) named positively and specifically, bulk and skin-adjacent wardrobe banned at build
  time. Sized to be selected from: at exp078's 26% review yield this gives ~50 usable triples
  against exp080's 34. 4 sharded jobs. Feeds a successor to exp108.
---
# exp109 — gen4 nudity dataset, realistic wardrobe

## Why gen1-gen3 have to be replaced rather than extended

Human review on 2026-08-11 found the clothing in the *edited* (training-target) clips reads wrong in
two distinct ways: implausibly baggy, or close enough to skin tone to pass as bare. Both are prompt
design, not generation luck.

Every gen1-gen3 `prompt_b` was written to satisfy the phrase **"no bare skin visible"**. The cheapest
way for a model to satisfy that constraint is a shapeless sack, and that is what the set asks for:

| gen1-gen3 wardrobe | problem |
|---|---|
| "a long floor-length robe" | bulk |
| "a heavy wool overcoat and thick trousers" | bulk |
| "a zipped-up parka" | bulk |
| "a thick turtleneck sweater" | bulk |
| "a thick heavy winter coat and trousers" | bulk |
| "wrapped in white towels" (gen3) | skin-adjacent |
| "matching black leotards" (gen3) | skin-adjacent |

Towels and leotards are the exact wardrobe [exp104](../exp104_clothed_retention_precompute/notes.md)
banned from the retention set for the same reason. **This is the same shape of mistake as exp079's
retention set**: a proxy constraint ("no bare skin") quietly determined the composition, and nobody
audited what it selected for until someone watched the clips.

## What gen4 changes

`tools/build_split_nudity_gen4_prompts.py` -> `prompts/split_nudity_gen4.csv`:

1. **Garments named positively and specifically** — "a fitted navy cotton t-shirt and straight-leg
   blue jeans" — with the "no bare skin visible" phrasing removed entirely. Coverage comes from
   naming something that covers, which is what a person would actually wear.
2. **Nothing bulky.** Outerwear is fitted (trench, tailored wool jacket, leather jacket); no robes,
   parkas, overcoats or heavy winter coats.
3. **Colour explicit and varied**, so the target distribution is not uniformly low-chroma.
4. **Banned at build time** and enforced by `verify()`: towels, robes, leotards, singlets, sports
   bras, swimwear, sleepwear, bare midriffs.

Scene grammar is held to what already generates: locked static cameras, exp078's close-up and
multi-person framings, one to three subjects. 25 scenes x 8 categories, each scene appearing once per
category, so a reviewer compares wardrobes against a fixed background instead of against noise.

**Not a colour fix.** Measured colorfulness of exp078's edited clips is 30.28 against 33.19 for the
originals — only -8.8%, while the trained model drops ~40% on nudity prompts. Wardrobe is not what
desaturates our output; eta=2 extrapolating past the donor is. This generation is about realism.

## The second defect it fixes

4 of exp080's 34 training targets **still trigger the nudity detector on the edited clip**:

| conf | frames | clip |
|---|---|---|
| 0.788 | 49/49 | exp078 run_005 `p25_s3511_edited.mp4` |
| 0.721 | 49/49 | exp078 run_005 `p28_s3514_edited.mp4` |
| 0.310 | 3/49 | exp061 `outputs_20260802_223148/p9_s3125_edited.mp4` |
| 0.304 | 1/49 | exp078 run_005 `p47_s3618_edited.mp4` |

The top two are two-person shots where the frame swap evidently covered only one body, and they are
full-strength across every frame. Those examples teach the model to answer a nudity prompt *with
nudity* — 6% of a 34-example set carrying the exact opposite of the objective.

**So review must check the edited clip for residual concept, not only for wardrobe realism.**
`edited_max_confidence` in the output metadata is a cheap first pass and should be sorted descending
before anyone watches anything; it would have caught all four.

## Why 200

This set is built to be **selected from**. exp078 kept 13 of 50 (26%) after review — split-prompt
triples have an A/B/C seam to heal and fail often, unlike exp104's plain single-prompt retention
clips which yielded 97.5%. At 26%, 200 triples give ~50 usable ones, which would be the first time
the training set is not the binding constraint (exp080 trained on 34).

## Review rule

Filter **within each wardrobe category** and record surviving per-category counts here, exactly as
exp104 does, and use `tools/filter_retention_metadata.py`'s discipline: a category that collapses
means regenerate, not proceed. exp079 became 55% skin precisely by ranking the whole set on per-clip
quality and keeping the top N. Write the filtered metadata to the **experiment root**, never under
`outputs_*/`, which is gitignored and would never reach the cluster.

## Status
- [x] Prompt set built, verified (200 rows, 8x25 balanced, no banned wardrobe, seeds 3801-4000 with
      no collision against gen1-gen3's 3103-3755).
- [x] Sharded into 4 round-robin parts, each carrying all 8 categories.
- [ ] Submitted (4 jobs).
- [ ] Reviewed for wardrobe realism AND residual concept; `edited_max_confidence` sorted first.
- [ ] Per-category survival recorded; filtered metadata written to the experiment root.
- [ ] Merged into a training dataset and run against exp108's best weight.
