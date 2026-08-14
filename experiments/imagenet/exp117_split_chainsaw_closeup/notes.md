---
status: ready
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  Rebuild of exp066 changing only the prompts: A and B reframed close and object-dominant, with the
  class-identifying detail the eval prompts carry. exp066 run 2 screened 7/30, and 17 of the 30
  losses were clips where the base model never rendered a chain saw at all — a prompt problem, not a
  sampler one. exp116 made the same change for faces and went 30% -> 50-63%. Also emits the
  whole-clip variant, which doubles as the per-row diagnosis of whether prompt A can render the
  object at all. Not submitted yet.
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

## Status
- [ ] Submitted.
- [ ] Screened; `no-concept` count compared against exp066 run 2's 17.
- [ ] Survivor `concept_region` balance checked.
- [ ] Decision recorded: split target, whole-clip target, or both, for exp069.
