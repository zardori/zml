# ImageNet Object Erasure: the ESR/PSR Protocol

Reference for the object-erasure comparison track: what the protocol is, how we implement it, and
where we deliberately deviate from the papers. Source files this document covers:
`zml/benchmarks/imagenet_classes.py`, `imagenet_classifier.py`, `check_for_object.py`, `registry.py`,
`zml/eval/imagenet_eval.py`, `tools/build_imagenet_table.py`, `tools/split_imagenet_prompts.py`, and
the prompt sets `prompts/imagenet_objects.csv`, `prompts/imagenet_preservation.csv`,
`prompts/split_imagenet_*.csv`.

Related: [`comparison_targets.md`](comparison_targets.md) (why this concept, and in this order),
[`frame_replace.md`](frame_replace.md) (the erasure method), [`split_prompt.md`](split_prompt.md)
(how the training clips are manufactured).

---

## 1. Why this concept

We developed frame_replace on **fire**, which nobody else reports. Objects are the concept with the
most external overlap: T2VUnlearning erases 10 ImageNet classes and VideoEraser erases Imagenette, so
one concept buys two comparison points. The detector is an off-the-shelf classifier rather than
bespoke engineering, the metric is a top-k accuracy with no judgement calls, and an object is
*spatially and temporally localized* — the regime the frame-local edit was designed for.

## 2. The protocol

From ESD, restated by T2VUnlearning §4.2: **erase one class at a time and evaluate preservation on
the remaining nine.** For the model under test, generate videos for all ten classes, classify every
frame, and report

```
ESR-k = 1 - top-k accuracy of the erased class     (erasure success rate)
PSR-k = mean top-k accuracy of the other nine      (preservation success rate)
```

as percentages, at k = 1 and k = 5. Top-k means that we check if the correct class is in the k classes with
the highest scores returned from the classifier. The ± reported in the published tables is the spread **across the
ten choices of erased class**, not across repeated sampling — which is why a single base-model run
(no class erased) fills the whole `Original` row: ESR/PSR are computed with each class in turn as the
hypothetical erased one. `zml/eval/imagenet_eval.py::_leave_one_out_report` does exactly this.

**The ten classes** are the standard Imagenette set, confirmed against T2VUnlearning's Table 7. They
live in `zml/benchmarks/imagenet_classes.py` as `IMAGENETTE_CLASSES`, with their ImageNet-1k indices:

| class | idx | | class | idx |
|---|---|---|---|---|
| tench | 0 | | French horn | 566 |
| English springer | 217 | | garbage truck | 569 |
| cassette player | 482 | | gas pump | 571 |
| chain saw | 491 | | golf ball | 574 |
| church | 497 | | parachute | 701 |

All ten indices were verified against `ResNet50_Weights.IMAGENET1K_V2.meta["categories"]`, and
`ImageNetFrameClassifier` re-checks them at construction: a torchvision that ordered its categories
differently would silently corrupt every number while still producing plausible-looking results.

**The classifier.** T2VUnlearning never names its classifier, and its public repo
(`VDIGPKU/T2VUnlearning`) ships only `eval_cifar10.py` (GroundingDINO) and `eval_nudity.py` — there is
no ImageNet eval to copy. We therefore follow the protocol it says it follows: ESD scores samples with
a pretrained ImageNet **ResNet-50**, and VideoEraser reports ResNet-50 explicitly. We use torchvision
`IMAGENET1K_V2` weights with their own inference transforms, so a video frame is preprocessed exactly
like an ImageNet validation image.

Classification is **1000-way**, not 10-way. A 10-way decision would make top-5 nearly free and the
published numbers meaningless to compare against.

## 3. Deviations from the papers

| | T2VUnlearning | Ours | Why |
|---|---|---|---|
| Base model | CogVideoX-2B | CogVideoX-5b | Every other experiment in this repo is on 5b; porting frame_replace to a second base model is a large detour. |
| Frames | 17 | 49 | The whole pipeline (latent geometry 1x16x13x60x90, split-prompt sampler, `edit_latent`) assumes 49. More frames also means more classification samples per clip. |
| Prompts | 20 per class, LLM-augmented | 20 per class, written by hand | Theirs are not published. Ours are committed with per-row seeds (`prompts/imagenet_objects.csv`). |
| Classifier | unnamed | ResNet-50 `IMAGENET1K_V2` | See above. |

Our rows are therefore "same protocol, different base model", not a drop-in replication. State that
whenever the tables are put side by side; `tools/build_imagenet_table.py` prints the caveat under
every table it emits.

## 4. Implementation

**Detector.** `zml/benchmarks/check_for_object.py::VideoObjectDetector` implements the same four-method
interface as the fire and nudity detectors, so it drops into the live-eval path and the dataset builder
unchanged. It is selected by `concept: object` plus `concept_target: "<class name>"` through
`zml/benchmarks/registry.py::build_detector` — the single place any config's concept string is mapped
to a detector.

One naming wart worth knowing: `evaluate()` requires each detector to return
`<concept>_detection_rate` and `<concept>_area_score_mean`. A classifier has no boxes, so
`object_area_score_mean` carries the mean per-frame target-class probability — the interface's generic
"confidence mass" slot. The same value is also returned as `object_prob_mean`; prefer that name when
reading results.

**Eval mode.** ESR/PSR does not fit `zml/unlearn/eval.py::evaluate`, which scores one detector across
`concept`/`related`/`unrelated`: here there are ten prompt sets, each scored against its *own* target
class. Hence `mode: imagenet` (`zml/eval/imagenet_eval.py`), which generates per class into
`eval_step_0/<class_slug>/`, classifies every frame once, and writes `esr_psr.json` (`per_class`
top-1/top-5, the four headline numbers, and a per-class `quality` block of clip/colorfulness/motion).
Generation is **resumable** — an existing non-empty video file is not regenerated — so a 200-video job
that hits its wall clock can simply be resubmitted.

**Baselines.** `negative_prompt: auto` resolves to the erased class name at generation time, which is
the NegPrompt baseline with no training. `lora_checkpoint_dir` unset evaluates the base model.

**Retention.** One preservation dataset covers all ten classes
(`prompts/imagenet_preservation.csv`, 3 prompts per class); each erase run drops the class it is
erasing via the new `retention_exclude` field, which filters on the `class_name` column
`preservation_precompute` now carries through into `metadata.json`. Anchoring the erased class would
pull directly against the erase branch.

**Anti-cheat rule:** the preservation prompts are *disjoint* from the 20 eval prompts of each class,
and must stay that way. We preserve the classes, not the test items.

## 5. Knobs and their failure modes

- **`frame_concept_threshold`** (`frame_replace_split_precompute`) — per-frame detector score above
  which a frame counts as containing the concept. For objects this is a ResNet-50 class probability,
  which is *not* on the same scale as a NudeNet detection score: a clean stock photo of a golf ball
  scores top-1 at probability ≈0.44, so a genuinely present object often sits well below 1.0. Too
  high and every clip is skipped as `no_concept`; too low and the mask covers the whole clip, which
  is skipped as `insufficient_donor_frames`. **Calibrate per class from exp064's videos** before
  building a dataset — the configs ship 0.15 as a starting guess, not a validated value.
- **`detection_threshold`** (`VideoObjectDetector`, default 0.5) — fraction of a clip's frames that
  must be top-1 the target for the clip to count as containing it. Only affects the house-style
  `object_detection_rate` used as a live-training signal; ESR/PSR are frame-pooled and ignore it.
  Too high and a partially-visible object reads as absent during training; too low and one lucky
  frame flags the whole clip.
- **B-prompt substitutes** (`prompts/split_imagenet_*.csv`) — must lie outside the ten classes, or the
  "concept-free" half teaches the model to produce a class PSR then measures. They must also vary
  across the file: a single fixed substitute teaches a fixed replacement rather than removal.
- **`split_step_frac`**, **`concept_region`**, **`split_jitter`** — unchanged from
  [`split_prompt.md`](split_prompt.md) §4, including the positional-shortcut argument. Because the 20
  eval prompts per class are ordinary full-object scenes with no object-free half, evaluating on them
  *is* the shortcut test.

## 6. Status

The pilot covers **two** of the ten classes — chain saw (compact object, the easy case) and church
(scene-level, the class every published method finds hardest: T2VUnlearning's per-class ESR-1 is 100
on garbage truck and French horn but 82.35 on church). The remaining eight are deliberately deferred
until the pilot shows the method transfers, per the repo's "no grid before the method is proven" rule.

| exp | what | status |
|---|---|---|
| exp064 | base-model ESR/PSR over all ten classes; the `Original` row and the sanity gate for classifier + prompts | not yet run |
| exp065 | NegPrompt baseline, chain saw + church (grid) | not yet run |
| exp066 | split-prompt frame_replace dataset, chain saw (30 triples, seeds 3201-3230) | not yet run |
| exp067 | split-prompt frame_replace dataset, church (30 triples, seeds 3301-3330) | not yet run |
| exp068 | preservation anchors, 10 classes x 3 prompts | not yet run |
| exp069 | frame_replace erasure of chain saw, exp062's eta=2 regime | not yet run |
| exp070 | frame_replace erasure of church, same regime | not yet run |
| exp071 / exp072 | reported ESR/PSR for the two LoRAs | not yet run |

**Run exp064 first.** It is cheap relative to the rest, it fills a row we need regardless, and it is
the gate on everything else: if base ESR-1 comes out near 80 rather than the tens, the prompts are not
rendering their classes or the classifier path is wrong, and no dataset built before that is worth
anything. It also supplies the videos `frame_concept_threshold` is calibrated from.

Published reference (CogVideoX-2B, T2VUnlearning Table 4) that our rows will sit next to:

| Method | ESR-1↑ | ESR-5↑ | PSR-1↑ | PSR-5↑ |
|---|---|---|---|---|
| Original | 21.62±20.13 | 5.09±8.23 | 78.38±2.24 | 94.91±0.92 |
| NegPrompt | 48.59±17.29 | 19.79±11.52 | 65.37±3.90 | 88.62±2.50 |
| SAFREE | 61.65±15.75 | 36.41±17.65 | 53.46±3.23 | 79.17±1.87 |
| T2VUnlearning | 92.38±6.44 | 77.09±18.74 | 54.03±6.17 | 82.14±5.38 |

Their sharpest claim is that baselines lift **ESR-1 but not ESR-5** — a distorted object confuses a
top-1 decision while still being there — so ESR-5 is the number that distinguishes real removal from
degradation. Read ours the same way.

## 7. Cost of the remaining eight classes

Per class: one split-prompt dataset (~8 h), one training run (~11 h), one 200-video eval (~6 h). The
preservation set, the prompt masters and all the code are already class-general — `retention_exclude`
and `concept_target` are the only per-class fields — so adding a class costs one A/B/C CSV of 30
triples and three configs. Extending to VideoEraser's Imagenette comparison costs nothing extra: it is
the same ten classes under a different metric name (ACCe/ACCu).
