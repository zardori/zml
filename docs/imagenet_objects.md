# ImageNet Object Erasure: the ESR/PSR Protocol

Reference for the object-erasure comparison track: what the protocol is, how we implement it, and
where we deliberately deviate from the papers. Source files this document covers:
`zml/benchmarks/imagenet_classes.py`, `imagenet_classifier.py`, `check_for_object.py`, `registry.py`,
`zml/eval/imagenet_eval.py`, `tools/build_imagenet_table.py`, `tools/split_imagenet_prompts.py`, and
the prompt sets `prompts/imagenet_objects.csv`, `prompts/imagenet_preservation.csv`,
`prompts/imagenet_objects/split/*.csv`.

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

  **Since `543eed8` this field is logging-only and gates nothing.** The concept mask is derived from
  `(split_latent_frame, concept_region)`, so the threshold decides only `concept_pixel_mask` in
  `metadata.json`, which nothing reads. It still earns its calibration, because the raw
  `frame_confidences` logged next to it are what `tools/screen_split_dataset.py` screens on after the
  build — the one job the detector still has here (§6).

  Before that commit the errors were asymmetric and dangerous: a frame *below* threshold became a
  donor, so a false negative silently spliced the object into `x0_edited`. exp066/exp067's run 1 is the
  cautionary record of what that cost — see §6.

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
- **A-prompt framing and specificity** (`prompts/imagenet_objects/split/*_closeup.csv`, built by
  `tools/build_split_imagenet_closeup_prompts.py`) — **the knob with the largest measured effect on
  yield, and the one that was wrong for the whole first pass.** Prompt A must put the object where a
  whole-frame classifier can see it, and must name the parts that identify the class. Too wide or too
  bare and the base model renders no object at all (17 of 30 rows in each of exp066/exp067); too
  tightly cropped and the clip stops resembling the eval prompts the erasure is measured on. Calibrate
  against `prompts/imagenet_objects.csv`, the set the base model scores 0.506 / 0.739 top-1 on.
- **B-prompt substitutes** (`prompts/imagenet_objects/split/*.csv`) — must lie outside the ten classes, or the
  "concept-free" half teaches the model to produce a class PSR then measures. They must also vary
  across the file: a single fixed substitute teaches a fixed replacement rather than removal. And they
  must be **specified as richly as A, and specified to lack the identifying feature**. Both halves of
  that matter: under-specified substitutes lose the splice on prompt strength rather than on content
  (which fakes yield by turning the safe half into the concept half), and church-shaped substitutes
  make the safe half score as `church` anyway — 10 of exp067's 30 rows, with `p22_s3323` at 0.2465
  concept-half against 0.2474 safe-half.
- **`--min-concept-max` / `--min-contrast-index`** (`tools/screen_split_dataset.py`) — the post-build
  screen, and the only place a detector is allowed to decide anything for these datasets. Too strict
  and a class with a small in-frame concept is thrown away wholesale; too loose and rows where the
  object was never rendered dilute the erase signal with a no-op ("replace non-chainsaw frames with
  other non-chainsaw frames"). 0.10 / 0.4 for both pilot classes; see
  [`split_prompt.md`](split_prompt.md) §3.1 for why the second gate is a within-clip differential.
- **`--max-degenerate-frac`** (same tool) — the third gate, added 2026-08-16, and the only one that
  looks at the *edited target* rather than the source clip. A blank frame scores p(concept) ≈ 0 like
  any concept-free frame, so a clip whose safe half never rendered gets a **perfect** contrast index,
  passes, and is then edited by mirroring that blank half into the concept region. exp122's
  `p22_s3353` scored +0.994 with a 49/49-blank target; exp118's `p4_s3305` was 73% blank *and* still
  showed a church, and it trained in exp070. Default 0.1 of frames; needs the videos beside the
  metadata and warns loudly when they are missing. Church hit this twice, chain saw never — expect it
  on classes whose scenes have large bright backgrounds.
- **`split_step_frac`**, **`concept_region`**, **`split_jitter`** — unchanged from
  [`split_prompt.md`](split_prompt.md) §4, including the positional-shortcut argument. Because the 20
  eval prompts per class are ordinary full-object scenes with no object-free half, evaluating on them
  *is* the shortcut test. Note that `split_step_frac` is **inert above ~0.5** (exp099); do not spend a
  run tuning it.
- **`tail_prompt_mode`** (`c` | `empty`) — what conditions the heal phase. **Measured dead (exp119):**
  at `split_step_frac` 0.85 the two modes give identical clips row-for-row, and at 0.3 `empty` was
  *worse* (2/5 against `c`'s 3/5), so prompt C's concept-deleting content is not why an early split
  loses the concept. Keep `c`, keep 0.85, and do not sweep this. `docs/split_prompt.md` §2.
- **`concept_guidance_scale`** — CFG on the concept branch (`pred_a`) only; `None` reuses
  `guidance_scale`. **Measured and rejected as a yield lever (exp120):** pass counts on the 12
  suppressed rows are 0/12 at 6.0 (the control, exactly as predicted), 2/12 at 9.0, 3/12 at 12.0 —
  against a pre-registered gate of 6/12. Leave it at `None`. What it *did* establish is the mechanism:
  at 9.0 **seven** of the twelve render the object again, and five of those seven then have it in both
  halves, so stronger A-conditioning defeats the suppression and loses the localization. Per-row
  response is also non-monotone in the scale (rows that render at 9.0 vanish at 12.0), so part of the
  effect is re-rolling the sample rather than strengthening the concept.
- **`split_mode`** (`prediction` | `trajectory`) — where the two prompts are combined during the
  split phase, and the successor to the knob above. `prediction` (the default, and every dataset up
  to exp122) keeps one latent and splices the prediction, which is what puts `pred_a` in a
  B-converging context in the first place; `trajectory` denoises each prompt on its own latent from
  shared noise and splices once at `split_step`. Same cost either way. exp127 measures it, with the
  6 currently-passing rows in its CSV as the coherence regression — independent trajectories share
  less context, and coherence across the seam is the thing that could pay for the yield.

## 6. Status

The pilot covers **two** of the ten classes — chain saw (compact object, the easy case) and church
(scene-level, the class every published method finds hardest: T2VUnlearning's per-class ESR-1 is 100
on garbage truck and French horn but 82.35 on church). The remaining eight are deliberately deferred
until the pilot shows the method transfers, per the repo's "no grid before the method is proven" rule.

| exp | what | status |
|---|---|---|
| exp064 | base-model ESR/PSR over all ten classes; the `Original` row and the sanity gate for classifier + prompts | **done** — gate passed, see below |
| exp065 | NegPrompt baseline, chain saw + church (grid) | **done** — and it splits by convention: 1000-way ESR-1 70.8 / 75.1 looks strong, restricted it is 17.1 / 0.2 with ESR-5 **0.00** for both. Most of its erasure is sibling confusion; report the restricted column |
| exp066 | split-prompt frame_replace dataset, chain saw (30 triples, seeds 3201-3230) | run 1 kept 4/30; run 2 (`0.85`, construction mask) kept 30/30 but **screens at 7/30** — superseded by exp117 |
| exp067 | split-prompt frame_replace dataset, church (30 triples, seeds 3301-3330) | run 1 kept 7/30; run 2 kept 30/30 but **screens at 3/30** — superseded by exp118 |
| exp068 | preservation anchors, 10 classes x 3 prompts | **done** — 30 entries, 3 per class, `outputs_20260803_233647` |
| exp099 | static vs motion-carrying A/B prompts x `split_step_frac` | **done** — motion prompts 0/5 two-state vs static 2/5; keep the static scaffold. Also showed `split_step_frac` is inert above ~0.5 |
| exp117 | chain-saw dataset on object-dominant prompts, `emit_whole_clip_target` | **done** — 14/30 usable (was 7/30); moved the thread's diagnosis, see below |
| exp118 | church dataset on object-dominant prompts, `emit_whole_clip_target` | **done** — 14/30 usable (was 3/30); survivors skew 10 first / 4 second |
| exp119 | `tail_prompt_mode` [c, empty] x `split_step_frac` [0.3, 0.85], 5 chain-saw seeds | **done** — hypothesis rejected; the tail is not a lever. `docs/split_prompt.md` §2 |
| exp069 | frame_replace erasure of chain saw, exp062's eta=2 regime | **done — the pilot's positive result.** top-1 0.506 → 0.00 from step 200, semantically (see below) |
| exp070 | frame_replace erasure of church, same regime | **done — negative.** Never erased; top-1 oscillated 0.00/0.32/0.00/0.22/0.47 and trended back to base |
| exp120 | `concept_guidance_scale` [6, 9, 12] on the 12 suppressed chain-saw rows | **done** — 0/12 → 2/12 → 3/12, gate not met; knob rejected, mechanism confirmed (§5) |
| exp121 / exp122 | gen2 datasets: exp117/exp118 prompts under fresh seeds, ~14 more rows each | **done** — 12/30 and 14/30; seed control passes, and exp122 fixed church's region skew to 7/7 |
| exp071 | reported ESR/PSR for the chain-saw LoRA | ready — wired to exp069's step-600 checkpoint |
| exp072 | reported ESR/PSR for the church LoRA | **blocked**, and deliberately so: exp070 has no checkpoint worth 200 videos |
| exp126 | `erase_esd_eta` [1.0, 1.5, 2.0] on the 33-row chain-saw merge | ready — attacks the freeze, see below |
| exp127 | `split_mode` [prediction, trajectory] on exp120's 12 rows + 6 survivors | ready — exp120's prescribed follow-up |
| exp128 | church rebuilt on the 27-row exp118+exp122 merge | ready, blocked on exp126 for the eta |

### The pilot's finding: erasure transfers, and it is concept-dependent

exp069 and exp070 are the same recipe — same eta, retention anchors, LR and step budget — differing
only in the concept and its dataset. **Chain saw erases and church does not.**

Chain saw goes to top-1 0.00 from step 200 through 600 on the 20 full-object eval prompts. Those
prompts have no object-free half, so the positional shortcut cannot explain it, and the frames agree
with the classifier: the workbench, plank, tool rack and lighting survive and the saw is replaced by
an unidentifiable plastic form. Preservation holds qualitatively (the other nine classes still render
correctly) and clip score stays at base.

Church never holds a zero for two consecutive checkpoints, and its top-5 climbs to 0.88 against a base
of 0.95. Two of its three candidate causes are data artefacts that exp128 repairs (14 rows skewed
10/4, one of them a 73%-blank target that still contained a church); the third is the concept itself —
removing a frame-filling structure means redrawing the frame, where a chain saw can be swapped inside
an untouched scene. This is what [`comparison_targets.md`](comparison_targets.md) §2.2 predicted, now
half-measured: exp128 decides whether the prediction or the dataset explains exp070.

### The defect that blocks the chain-saw row: the concept clips freeze

exp069's erasure comes with a **"static poster"** signature on the concept set: motion 0.010 against a
base of 0.564 (−98%), present already at step 100, with colorfulness +40% and clip score unchanged.
The clips are still images with boosted saturation.

Two properties make this its own failure mode rather than a repeat of nudity's:

- **It is concept-conditional.** The unrelated set loses 30% of its motion, the concept set 98%. In
  nudity, exp107 located the motion collapse as a *global* property of the adapter.
- **DOVER cannot see it.** Technical scores are 0.084 (concept) and 0.078 (unrelated) against a base
  of 0.100 — no separation. DOVER measures spatial/technical quality, not temporal liveness, so on
  this failure `motion_score` is the instrument. (`imagenet_eval` now records DOVER in its per-class
  `quality` block, and `--rescore` backfills it on any x86_64 machine; helios omits the keys rather
  than writing 0.0.)

exp126 sweeps `erase_esd_eta` to separate the two readings — the erase pressure is high enough that
freezing is the cheapest way to satisfy it, versus the LoRA having learned "chain-saw prompt → still
life". Motion rising with erasure intact means the former; motion and top-1 rising together at every
setting means the latter, and the fix moves to the retention branch.

**exp117/exp118 — the prompt reframe worked, and then changed the question.** Both classes went to
14/30 usable, and the reframe is the measured cause: same seeds, same sampler, only A and B rewritten.
Church's second failure has its own confirmed cure — its substitutes now peak at p(church) 0.064
where exp067's tied the concept half at 0.247.

The more consequential result is what `emit_whole_clip_target` reported. **Plain prompt A renders the
object in 29/30 chain-saw and 28/30 church rows**, so the exp066/exp067 story — the base model never
drew it — is essentially closed. What is left is the *splice* suppressing a concept the identical
(prompt, seed) renders fine, failing binary: surviving rows keep 1.12x the plain-A confidence, failing
rows 0.06x. Mechanism, the `concept_guidance_scale` response, and why the whole-clip pairs must not be
used as training targets (same-seed A and B differ nearly as much as unrelated scenes):
`docs/split_prompt.md` §3.3–3.4.

Two operational notes from those builds. Screened keep-lists live at the experiment root as
`outputs_{timestamp}_screened.json` — `outputs_*/` is gitignored, so anything the cluster must read
has to sit outside it. And exp118 produced two degenerate near-white clips (`p9_s3310` std 0.0,
`p21_s3322` std 11.8); both fell out on the concept screen, but `p21_s3322` scored 0.445 on whole-clip
A, so a whole-clip-based keep rule would have admitted a blank video.

**exp066/exp067 run 1 — why the datasets were discarded.** Both ran 2026-08-03, two days before
`543eed8` made the concept mask construction-derived, so their masks were still detection-derived.
Yields were 4/30 and 7/30, and in both the skip reasons are perfectly bimodal — every `no_concept` row
has 13 donor frames and every `insufficient_donor_frames` row has 0. That is a detector reading noise
and splitting it, not a detector finding an object and missing some frames. Church made it explicit:
six of its seven *kept* rows have confidences that never leave the 0.021–0.050 band against a 0.03
threshold, and four have `edited_max_confidence` at or above threshold, meaning the edit removed
nothing by the detector's own reading. All seven kept church rows were `concept_region: first` despite
`concept_region: random`, which alone would have taught the positional shortcut.

**exp066/exp067 run 2 — the rebuild worked and the yield did not move.** With the mask
construction-derived, `insufficient_donor_frames` geometric and `split_jitter: 1`, both runs kept
30/30 as predicted. Screening them (`tools/screen_split_dataset.py`) shows what the keep count hides:

| | rows | pass | `not-split` | `no-concept` |
|---|---|---|---|---|
| exp066 chain saw | 30 | 7 (23%) | 6 | **17** |
| exp067 church | 30 | 3 (10%) | 10 | **17** |

`no-concept` means the peak detector score in the concept half never reached 0.10 — the base model
drew no chain saw, no church, anywhere in the clip. **The same 17 of 30 in both classes.** A splitter
cannot separate a concept that was never rendered, so most of what looked like a sampler problem never
was one.

Three things follow, and together they redirect the thread:

1. **`split_step_frac` was the wrong lever, and is now measured to be a dead one.** Raising it 0.5 →
   0.85 was justified by the argument that prompt C deletes the object for these classes (it does).
   But exp099 ran the same seeds at both values and got near-identical clips — 2–4 grey levels apart,
   every verdict unchanged — because content is committed in roughly the first 20 of 50 steps and a
   switch after that only refines. Anything in [0.5, 1.0] is the same experiment. The C-deletion
   argument survives, but it applies to the *decisive* window below ~0.4, which is what exp119 tests
   with `tail_prompt_mode: "empty"` (a tail conditioned on nothing, so it heals without erasing).
2. **The lever that moves object yield is the prompts.** exp116 proved this on faces: reframing to
   controlled medium/close framing took yield 30% → 50–63%, while a re-seed of the original prompts
   reproduced 30% exactly. The object prompts have the same two defects — wide framing that puts a
   small object in a large scene, and no class-identifying detail ("a church", against the eval set's
   "a stone church with a tall steeple"). exp117/exp118 rebuild them with
   `tools/build_split_imagenet_closeup_prompts.py`, settings, seeds and prompt C held verbatim.
3. **Church has a second, class-specific failure.** Its 10 `not-split` rows are ones where the church
   *was* rendered and the substitute building scored just as high (`p22_s3323`: 0.2465 concept half,
   0.2474 safe half). "a village hall", "a museum facade", "a manor house" are masonry buildings of
   similar scale, and ResNet-50's `church` class is not narrow. exp118's substitutes are specified to
   have no tower, spire or bell-cote at all.

**Screen these clips with the detector differential, not with seam contrast.** The pixel-space checker
is concept-blind and whole-frame, and on this data it makes both errors: it passes exp066's
`p13_s3214` (flower pot → chain-and-hook object, a textbook seam, peak p(chain saw) 0.003) and rejects
exp067's `p27_s3328` (bell tower present for 24 frames then gone — the one correct church split —
scored "diffuse" at ratio 3.0, because a bell tower is a small share of the frame). Use
`tools/screen_split_dataset.py`, which asks the paired within-clip question instead; keep
`tools/check_seam_contrast.py` for diagnosing *why* a row failed. `docs/split_prompt.md` §3.1.

**exp064 (done) — the gate, and what it changed.** 200 videos in 5.71 h on athena. Both pilot classes
render well (chain saw top-1 .506 / top-5 .795, church .739 / .950), so the pilot is viable and the
datasets are worth building. It also produced the two things everything downstream needed: the
calibrated thresholds in §5, and the discovery that the ranking convention is ambiguous (§3.1). Full
numbers, per-class weak spots and the yield risks carried into exp066/exp069:
`experiments/imagenet/exp064_eval_base_imagenet/notes.md`.

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
