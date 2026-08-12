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
  time. DONE: 100 of 200 kept (50%), against exp078's 26% — the realism rewrite nearly doubled
  yield, and 100 targets is 3.2x exp080's filtered 31. Category survival 36-64%, mild enough that no
  gen5 rebalance is warranted. Detector confidence stayed uncorrelated with review. Feeds the two
  training runs that follow exp108.
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

| conf | frames | clip | human verdict (2026-08-11) |
|---|---|---|---|
| 0.788 | 49/49 | exp078 run_005 `p25_s3511_edited.mp4` | **reject** |
| 0.721 | 49/49 | exp078 run_005 `p28_s3514_edited.mp4` | **reject** |
| 0.310 | 3/49 | exp061 `outputs_20260802_223148/p9_s3125_edited.mp4` | **reject — worst of the four** |
| 0.304 | 1/49 | exp078 run_005 `p47_s3618_edited.mp4` | keep |

The top two are two-person shots where the frame swap evidently covered only one body, and they are
full-strength across every frame. Those examples teach the model to answer a nudity prompt *with
nudity* — and with the third, 9% of a 34-example set carried the exact opposite of the objective.
`experiments/exp080_frame_replace_nudity_gen2/metadata_human_filtered.json` is the surviving 31.

### The detector cannot do this review, and the ranking proves it

The obvious shortcut is to sort by `edited_max_confidence` and watch the top of the list. **That
would have failed here.** Human review called `s3125` the *worst* clip in the set, and the detector
ranked it third at 0.310 — statistically indistinguishable from `s3618` at 0.304, which was kept.

So the confidence score separates the 0.72-0.79 band (genuinely broken) from everything else, and
has **no discriminative power inside the 0.30 band**, where the worst clip and an acceptable one
differ by 0.006. Use it to catch the catastrophic cases cheaply, then watch every clip anyway — this
is [[feedback-detector-metrics-not-ground-truth]] with an unusually clean margin, and at 200 triples
the temptation to shortcut will be much stronger than it was at 34.

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

## Results (2026-08-11) — 100 of 200 kept

| run | shard | kept |
|---|---|---|
| run_001 | part1 | 28/50 (56%) |
| run_002 | part2 | 25/50 (50%) |
| run_003 | part3 | 20/50 (40%) |
| run_004 | part4 | 27/50 (54%) |
| **total** | | **100/200 (50%)** |

**50% against exp078's 26%** — the realistic-wardrobe prompts nearly doubled the yield, and 100
usable triples is 3.2x the 31 that survive in exp080's filtered dataset. For the first time the
training set is not the binding constraint.

Per-run filtered metadata sits at this experiment's root as `metadata_human_filtered_run00N.json`
(git-tracked; the `grid_*/run_*/outputs/` originals are gitignored and would never reach the cluster
— note `tools/filter_retention_metadata.py`'s default output path assumes the single-run
`outputs_*/` layout and lands *inside* the ignored tree for grid runs, so `--output` must be passed
explicitly here).

### Category survival — mild, and not worth acting on

| category | kept/gen | rate |
|---|---|---|
| formal_wear | 16/25 | 64% |
| outerwear | 15/25 | 60% |
| traditional | 14/25 | 56% |
| knitwear | 12/25 | 48% |
| workwear | 12/25 | 48% |
| summer_light | 12/25 | 48% |
| casual | 10/25 | 40% |
| uniform | 9/25 | 36% |

Structured garments (formal_wear, outerwear) hold the top two in all four runs, so there is a real
effect — a tailored suit edits in more cleanly than a t-shirt. But the spread is only 36-64%, nothing
collapsed, and the bottom of the ranking is unstable: after three runs casual looked like a clear
outlier at 32% and run_004 returned 4/6 for it, ending at 40% with uniform lowest. **No rebalanced
gen5 is warranted**, and the earlier "plain clothing is hardest" reading was over-claimed on n=19.

### The detector remains unusable on these targets

Across the four runs a large share of *accepted* clips trip `edited_max_confidence >= 0.2`, many at
49/49 frames — the same signature that marked exp080's genuinely broken targets. Human review
confirmed the gen4 ones are fine. The likeliest mechanism is that gen4 deliberately asks for
**fitted** clothing (bulk was banned by design), and fitted fabric on a close-up torso gives NudeNet
contours to fire on. Consequence: the score is *less* usable on gen4 than on gen1-gen3, and target
screening stays fully manual. Third independent confirmation of
[[feedback-detector-metrics-not-ground-truth]] in two days.

The `MIN_OVERALL_KEEP_FRACTION` guard in `tools/filter_retention_metadata.py` refused run_003 at 40%.
That threshold is calibrated for retention sets (exp104 yielded 97.5%), where a low keep rate means
the prompts are wrong; split-prompt triples fail far more often by construction. Overridden with
`--allow-skew` for run_003 only — a threshold mis-fit, not a signal about the review.

## Status
- [x] Prompt set built, verified (200 rows, 8x25 balanced, no banned wardrobe, seeds 3801-4000 with
      no collision against gen1-gen3's 3103-3755).
- [x] Sharded into 4 round-robin parts, each carrying all 8 categories.
- [x] Submitted and complete (4 jobs).
- [x] Reviewed clip by clip for wardrobe realism and residual concept; 100/200 kept.
- [x] Per-category survival recorded; filtered metadata at the experiment root.
- [ ] Merged into a training dataset (needs the cluster — latents live in four `run_*/outputs/latents`).
- [ ] Trained against exp108's best `retention_weight`, gen4-only and gen4+exp080-filtered.
