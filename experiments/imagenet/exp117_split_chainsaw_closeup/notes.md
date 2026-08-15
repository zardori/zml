---
status: done
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  The prompt reframe worked: 14/30 usable against exp066 run 2's 7/30, `no-concept` 17 -> 12,
  survivors balanced 6 first / 8 second. The whole-clip diagnostic then moved the diagnosis for the
  whole thread — plain prompt A renders a chain saw in 29 of 30 rows, so the remaining loss is the
  *splice* suppressing a concept the same (prompt, seed) renders fine, and it fails binary (surviving
  rows keep 112% of the plain-A confidence, failing rows 6%). exp120 attacks that; exp121 samples
  more rows. The whole-clip pairs are NOT a usable training target — same-seed A and B differ almost
  as much as unrelated scenes. Dataset for exp069.
---
# exp117 — split-prompt chain-saw dataset on object-dominant prompts

## Goal
Raise usable yield on the chain-saw dataset by fixing the prompts, having established that the
sampler is not what is losing the rows.

## Why the prompts
`tools/screen_split_dataset.py` on exp066 run 2 (`outputs_20260808_235138`) reports:

```
30 clips | pass 7 (23%) | not-split 6 | no-concept 17
```

`no-concept` means the peak p(chain saw) over all 49 frames never reached 0.10 — the base model drew
something else. Seventeen rows. That is not the splitter failing to separate a concept; there was no
concept to separate. exp067 (church) reports **the same 17**, which is a strong hint that the cause
is shared and structural rather than per-class bad luck.

Two differences from `prompts/imagenet_objects.csv`, the eval set the base model scores 0.506 top-1
on for this class:

1. **Framing.** "Static shot of a chain saw resting on a wooden workbench in a cluttered garage" —
   a small object in a cluttered wide frame. ResNet-50 classifies a 224px view of the whole frame, so
   a small object is not merely hard to detect, it is genuinely not what the frame is *of*.
2. **Specificity.** Eval prompts name the identifying parts ("its orange casing and bar clearly
   visible"); the split prompts said only "a chain saw".

`tools/build_split_imagenet_closeup_prompts.py` fixes both, and applies (2) **symmetrically to B** —
if only A gained detail, B would lose the splice on prompt strength rather than on content, which
would buy yield by quietly turning the safe half into the concept half.

Held fixed so the comparison isolates the prompts: all 30 settings and their seed order (same seed =
same scene as exp066 run 2), the substitute objects, prompt C verbatim, and every sampler knob. The
static-camera scaffold stays too — exp099 tested motion-carrying prompts and they were strictly worse
(0/5 two-state against 2/5).

## What `emit_whole_clip_target` buys here
Beyond the face thread's reason, it makes the next failure self-diagnosing. The A-side confidences
come from a *plain* generation of prompt A, so they say whether the base model can render a chain saw
for this (prompt, seed) at all, independent of the splice. If yield is still low, this run says which
of the two mechanisms to fix without spending a second job. It is also a seam-free fallback target:
A and B differ by one noun under one seed.

Cost: two extra plain generations per row, 92 -> 192 transformer forwards, ~2.1x. exp066 run 2 was
197 s/row on helios, so expect ~3 h 30 m against the 6 h limit.

## What to watch
- `tools/screen_split_dataset.py --metadata <outputs>/metadata.json --min-concept-max 0.10`. The
  number that matters is **`no-concept`**: it should fall well below 17. If it does not, the reframe
  failed and the next move is (prompt, seed) pre-screening, not more prompt editing.
- `concept_region` balance *among survivors*, not among all 30. exp067's survivors were 3 first / 0
  second, which is the positional shortcut waiting to happen; screening can concentrate a skew that
  the full set does not have.
- Whole-clip A-side vs split-clip confidences. A row where plain A renders the chain saw but the
  split clip does not is a genuine splitter failure and is the interesting case.

## Downstream
Replaces exp066 as exp069's dataset if it clears yield. Wire exp069's `metadata_file` / `latents_dir`
to this run's `outputs_{timestamp}` — and prefer the screened subset (`--write-filtered`) over the
raw 30.

## Results (`outputs_20260815_014333`, helios, 3 h 26 m, 30/30 rows kept, 0 skipped)

```
30 clips | pass 14 (47%) | not-split 4 | no-concept 12
surviving concept_region balance: 6 first / 8 second
--keep-seeds 3201 3202 3204 3209 3210 3213 3214 3216 3218 3219 3221 3225 3227 3229
```

Against exp066 run 2 (same seeds, same sampler, wide prompts): **7 → 14 pass, `no-concept` 17 → 12,
`not-split` 6 → 4**. The reframe delivered, at almost exactly exp116's face-thread magnitude. The
survivors are balanced across `concept_region`, so screening did not concentrate a positional skew.

Screened keep-list committed as `outputs_20260815_014333_screened.json` (the experiment root is not
gitignored, unlike `outputs_*/`).

Spot-checked by eye: `p1_s3202`, `p8_s3209`, `p0_s3201` are textbook two-state clips — background,
lighting and camera identical across the seam, chain saw in one region and a watering can / spade /
bicycle pump in the other, and `_edited.mp4` concept-free throughout.

## What `emit_whole_clip_target` actually told us — the thread's diagnosis has changed

This is the finding that outlives the dataset. Prompt A's *plain* clip clears p(chain saw) 0.10 in
**29 of 30 rows** (church: 28/30). The exp066/exp067 story — "the base model never drew the object" —
is essentially gone. What remains is the splice suppressing a concept the identical (prompt, seed)
renders fine unsplit, and it fails binary rather than gradually. Split concept-half mean over the
same row's plain prompt-A mean:

| rows | median ratio |
|---|---|
| passing | 1.12 |
| failing | **0.06** |

No middle. `generate_split_clip` predicts both branches over the whole latent and splices only the
prediction, so `pred_a` sees a context converging on prompt B and CogVideoX's temporal-coherence
prior drags the concept region to match. Either the object establishes itself early or it is gone.
**exp120** sweeps `concept_guidance_scale` against exactly this.

Weaker secondary signal: `concept_region: second` passed 8/13 against `first` 6/17. Not acted on —
fixing the side is the positional shortcut `random` exists to prevent.

## Do not train on the whole-clip variant

It was proposed as a seam-free fallback target on the argument that A and B differ by one noun under
one seed. Measured, that argument fails. Mean per-pixel |A − B| at the same seed is **56.5**, against
**72.4** between A clips of two *unrelated* rows, with 74% of pixels moving more than 25 levels.
The noun swap redraws the frame; the same seed does not hold the scene. Training on it would teach a
global scene substitution, which is the opposite of frame_replace's minimal-edit premise. Church is
milder (52.9 vs 86.4, 50% of pixels) and still not a controlled counterfactual.

So `emit_whole_clip_target` earned its 2.1x once, as a diagnostic. exp121/exp122 turn it off.

## Status
- [x] Submitted.
- [x] Screened; `no-concept` 12 against exp066 run 2's 17.
- [x] Survivor `concept_region` balance checked — 6 / 8, healthy.
- [x] Decision recorded: **split target only**. Whole-clip rejected on the pixel-difference
      measurement above.
