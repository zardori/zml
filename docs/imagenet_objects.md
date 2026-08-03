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

Every number is reported under **two ranking conventions**, because the papers do not say which they
used and the choice moves the numbers by ~20 points — see §3.1.

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

### 3.1 Two ranking conventions (1000-way and 10-way)

Top-k accuracy needs a candidate set, and neither paper states theirs. The two readings are:

- **1000-way** — rank over all of ImageNet-1k. Faithful to ESD's stated setup, and the stricter test
  of preservation, but it charges the model for *taxonomy* rather than *rendering*.
- **10-way (restricted)** — rank within the ten protocol classes only. Immune to sibling-class
  confusion, and a harder test of erasure: to raise ESR the model must render something that reads as
  one of the other nine, not merely something ResNet-50 is unsure about.

exp064 showed this is not a detail. Under 1000-way our base model's mean top-1 is 55.0%; restricted,
it is 90.1%. The gap is almost entirely three classes whose ImageNet neighbours are near-duplicates:
`cassette player` scores 7.4% top-1 while *cassette* (21.9%) and *tape player* (18.9%) take the mass,
`English springer` loses to *Gordon setter* / *English setter*, and `tench` to *barracouta* / *gar*.
The object is rendered correctly in each case; the classifier is splitting hairs that the protocol
does not care about. Better prompts cannot fix this.

**Which one did the papers use?** Probably something close to restricted. Re-scoring the same exp064
frames 10-way lands the top-5 numbers almost exactly on T2VUnlearning's `Original` row (ESR-5 3.44 vs
their 5.09, PSR-5 96.56 vs 94.91), where 1000-way is ~18 points adrift. Their PSR-5 of 94.91±0.92 is
also hard to believe as a 1000-way number on a 2B model when our stronger 5b manages 76.52. This is
inference, not evidence — hence reporting both rather than picking.

**How it is implemented.** `ImageNetFrameClassifier.probs()` runs the network once and
`topk_indices(probs, k, restrict_to)` ranks it; `restrict_to=IMAGENETTE_INDICES` maps back to
absolute ImageNet indices so callers never learn which convention produced a hit. One forward pass
serves both. In `esr_psr.json` the top level stays 1000-way and the restricted copy of the whole
report — `per_class`, `per_erased_class`, `mean`, `std` — is nested under `"restricted"`.
`tools/build_imagenet_table.py` prints one table per convention.

**Re-scoring is free.** `python -m zml.eval.imagenet_eval --rescore <outputs_dir> --prompts-csv <csv>`
reclassifies a finished run's videos without constructing the diffusion pipeline, so a metric change
costs minutes on a laptop instead of hours of cluster time. exp064's report was produced this way
after the fact.

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
  which is *not* on the same scale as a NudeNet detection score: church frames never exceed 0.49 even
  when the building fills the frame, so a nudity-style 0.5 would mask nothing at all.

  **The errors are not symmetric.** A frame *below* threshold becomes a donor
  (`frame_replace_split_precompute.py:133-135`), so a false negative silently splices the object into
  `x0_edited` and poisons the training target; a false positive only shrinks the donor pool and
  surfaces loudly as `insufficient_donor_frames` in the skip list. **Err low.**

  **Calibrate against the negative distribution**, not by eye: score the target class on the *other*
  nine classes' clips and put the threshold just above that ceiling. From exp064 (8820 negative
  frames, 980 positive per class):

  | class | negative p99.9 / max | positive p25 / p50 / p75 | threshold | TPR | FPR |
  |---|---|---|---|---|---|
  | chain saw | 0.018 / 0.044 | 0.024 / 0.115 / 0.385 | **0.05** | 64.7% | 0.0% |
  | church | 0.003 / 0.006 | 0.127 / 0.233 / 0.313 | **0.03** | ≥85% | 0.0% |

  Separation is wide enough that false positives cost nothing anywhere in this range — which is why
  the 0.15 the configs originally shipped was strictly worse than 0.05: it bought no precision and
  discarded a fifth of the real concept frames. Note the per-class spread (0.044 vs 0.006): do not
  assume one class's threshold transfers to another.
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
| exp064 | base-model ESR/PSR over all ten classes; the `Original` row and the sanity gate for classifier + prompts | **done** — gate passed, see below |
| exp065 | NegPrompt baseline, chain saw + church (grid) | not yet run |
| exp066 | split-prompt frame_replace dataset, chain saw (30 triples, seeds 3201-3230) | not yet run |
| exp067 | split-prompt frame_replace dataset, church (30 triples, seeds 3301-3330) | not yet run |
| exp068 | preservation anchors, 10 classes x 3 prompts | not yet run |
| exp069 | frame_replace erasure of chain saw, exp062's eta=2 regime | not yet run |
| exp070 | frame_replace erasure of church, same regime | not yet run |
| exp071 / exp072 | reported ESR/PSR for the two LoRAs | not yet run |

**exp064 (done) — the gate, and what it changed.** 200 videos in 5.71 h on athena. Both pilot classes
render well (chain saw top-1 .506 / top-5 .795, church .739 / .950), so the pilot is viable and the
datasets are worth building. It also produced the two things everything downstream needed: the
calibrated thresholds in §5, and the discovery that the ranking convention is ambiguous (§3.1). Full
numbers, per-class weak spots and the yield risks carried into exp066/exp069:
`experiments/exp064_eval_base_imagenet/notes.md`.

Our `Original` row, both conventions, against the published one:

| row | ESR-1↑ | ESR-5↑ | PSR-1↑ | PSR-5↑ |
|---|---|---|---|---|
| ours 5b/49f, 1000-way | 45.01 ± 25.22 | 23.90 ± 20.11 | 54.99 ± 2.80 | 76.10 ± 2.23 |
| ours 5b/49f, 10-way | 9.91 ± 9.57 | 3.44 ± 5.56 | 90.09 ± 1.06 | 96.56 ± 0.62 |
| T2VUnlearning 2B/17f | 21.62 ± 20.13 | 5.09 ± 8.23 | 78.38 ± 2.24 | 94.91 ± 0.92 |

Mean ESR-1 and mean PSR-1 sum to 100 by construction: averaged over all ten choices of erased class,
both collapse to the overall mean top-1 accuracy. An `Original` row carries two independent numbers,
not four — worth knowing before reading agreement into it.

**Reproducibility floor.** ResNet-50 inference is not bit-identical across GPUs: scoring exp064's
same video files on the A100 that generated them versus on a local card moved per-class top-1 by
~0.002 and ESR-5 by 0.42, as near-tie frames flip under different cuDNN kernels. Sub-1-point
differences between runs are noise; the effects this protocol is meant to detect are tens of points.

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
