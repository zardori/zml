---
status: done
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  THE PILOT'S POSITIVE RESULT: frame_replace erases an ImageNet object class, semantically. Concept
  top-1 0.506 -> 0.00 from step 200 onward on the 20 full chain-saw eval prompts, which have no
  object-free half, so the positional shortcut is ruled out; frames confirm the workshop scene
  survives and only the saw is replaced. The defect is a NEW failure mode: the concept clips freeze
  (motion 0.010 vs base 0.564, -98%) and over-saturate (colorfulness +40%) while clip score stays at
  base — a "static poster". Unlike nudity (exp107, global motion loss), the freeze is
  concept-conditional: the unrelated set only loses 30%. exp071 reports the real 200-prompt row;
  exp123 attacks the freeze via eta.
---
# exp069 — frame_replace erasure of "chain saw"

## Goal
The question the whole pilot exists to answer: does frame_replace erase an **ImageNet object** class,
and does it do so semantically rather than positionally? Chain saw is the easy half — a compact
object on a bench, which is the regime `docs/comparison_targets.md` §2.2 argues frame_replace was
designed for.

## Setup
Field-for-field identical to exp062 (nudity, eta=2) except the dataset, `concept`/`concept_target`,
and the retention set. Keeping the recipe fixed is the point: if chain saw erases and nudity did not,
the difference is the concept, not the hyperparameters.

- Dataset: exp117 + exp066 (split-prompt manufactured partial clips), screened and merged — below.
- Retention: exp068's ten-class anchors minus chain saw (`retention_exclude`).
- Regime: `erase_input_latent: original`, velocity loss, `erase_esd_eta: 2`, t in [400, 1000),
  constant LR 5e-4, 600 steps, rank-8 LoRA, `gradient_accumulation_steps: 4`.

## Dataset: 21 rows, merged from two builds

exp117 unblocked this. Its object-dominant prompts took chain-saw yield from 7/30 to 14/30, and the
config now trains on those 14 plus exp066 run 2's 7 screened survivors.

Merging rather than taking exp117 alone is a deliberate call, on two grounds:

- **Size.** exp062, the nudity run whose recipe this copies field-for-field, trained on 31. Fourteen
  is thin for 600 steps at rank 8.
- **Framing diversity, which is the more important one.** All 30 exp117 prompts share the closeup
  scaffold — "in close view, filling much of the frame", static camera. The 20 chain-saw eval prompts
  are ordinary scenes. A LoRA trained only on frame-filling objects has to generalise across framing
  to score on the eval set, and exp066's rows are the only wide-framing clips available.

The two sources also happen to balance the positional shortcut: 6 first / 8 second and 4 first /
3 second merge to 10 / 11.

**Before submitting**, build the merged dataset on the cluster — `combined_dataset/` is gitignored
and the `.pt` latents only exist there:

```
./merge_dataset.sh --cluster helios \
  --output experiments/imagenet/exp069_frame_replace_chainsaw/combined_dataset \
  --source experiments/imagenet/exp117_split_chainsaw_closeup/outputs_20260815_014333_screened.json \
           experiments/imagenet/exp117_split_chainsaw_closeup/outputs_20260815_014333/latents \
  --source experiments/imagenet/exp066_split_chainsaw_dataset/outputs_20260808_235138_screened.json \
           experiments/imagenet/exp066_split_chainsaw_dataset/outputs_20260808_235138/latents
```

Then `./submit_job.py helios experiments/imagenet/exp069_frame_replace_chainsaw/config.yaml`.

## What to watch
Live eval writes `summary.json` every `save_interval`; read that first.
- **Erasure:** `concept_detection_rate` on the concept set should fall well below exp064's base level
  for chain saw.
- **Shortcut test:** the concept prompts are ordinary full chain-saw scenes with no object-free half.
  A drop there means the LoRA learned to remove the object, not to copy the clean half of a training
  clip.
- **Collateral:** the unrelated set (one prompt per preserved class) should hold its detection rate,
  clip score and motion near base. A collapse there is the PSR failure exp071 would confirm.
- Watch for overfitting: 21 rows against 600 steps. If the erase loss floors early while the eval
  barely moves, that is memorisation of 21 clips, and the answer is exp121's rows, not more steps.
- **Framing generalisation.** Two thirds of the training rows are frame-filling closeups; every eval
  prompt is an ordinary scene. If the concept set barely moves while training looks healthy, check a
  few eval videos before concluding the method failed — erasing only at closeup framing is a
  different (and more informative) failure than not erasing.

## Results (`outputs_20260816_003333`, helios, 11.1 h, 600/600 steps, exit 0)

Live monitor, n=9 concept prompts and 9 unrelated. Base row is exp064's chain-saw class.

| step | top-1 | top-5 | clip | colorfulness | motion | DOVER tech | unrel. motion |
|---|---|---|---|---|---|---|---|
| base | 0.506 | 0.795 | 0.322 | 49.4 | 0.564 | — | ~0.66 (9-class mean) |
| 100 | 0.11 | 0.25 | 0.30 | 60 | 0.040 | 0.077 | 0.38 |
| 200 | **0.00** | 0.17 | 0.32 | 61 | 0.010 | 0.089 | 0.54 |
| 300 | **0.00** | 0.00 | 0.28 | 73 | 0.010 | 0.084 | 0.51 |
| 400 | **0.00** | 0.11 | 0.30 | 63 | 0.030 | 0.090 | 0.45 |
| 500 | **0.00** | 0.39 | 0.31 | 66 | 0.020 | 0.101 | 0.46 |
| 600 | **0.00** | 0.27 | 0.32 | 69 | 0.010 | 0.084 | 0.46 |

DOVER was backfilled locally with `tools/score_dover.py` — helios writes 0.0 there (aarch64).

**1. The erasure is real, and it is semantic.** Top-1 is 0.00 from step 200 on, and the eval prompts
are ordinary full chain-saw scenes with no object-free half — so a LoRA that had only learned the
positional rule "copy the clean half onto the other" could not have moved them. Frames from
`eval_step_600/concept/video_0.mp4` show why the classifier is right: the workbench, the plank, the
tool rack and the lighting are all intact, and where the saw was there is an orange-and-blue plastic
form with no bar, no chain and no teeth. Removal, not scene destruction. **This is the answer the
object pilot was set up to get.**

**2. Preservation holds, qualitatively.** Unrelated clips (cassette player on a desk, French horn on
a chair) render correctly and stay recognizable at step 600. Concept clip score is at base (0.32 vs
0.322), so the clips still match their prompts as scenes.

**3. The defect is a "static poster" signature, and it is new.** Concept motion is 0.010 against a
base of 0.564 — a 98% loss, present already at step 100 — with colorfulness up 40% (69 vs 49) and
clip score flat. Every concept clip is effectively a still image with boosted saturation. DOVER
technical reads 0.084 on the concept set and 0.078 on the unrelated one, against a base of 0.100 for
this class: both sets sit ~16-22% below base and **DOVER does not separate the frozen concept clips
from the healthy unrelated ones at all.** That is consistent with these being clean *stills* rather
than broken video, and it is the caveat to carry: DOVER measures spatial/technical quality, not
temporal liveness, so on this failure mode `motion_score` is the instrument and DOVER is not.

**4. The freeze is concept-conditional, which differs from nudity.** Unrelated motion is 0.46 against
a base class mean of ~0.66 (−30%), where the concept set loses 98%. exp107 located nudity's motion
collapse as a *global* property of the adapter; here the adapter mostly damages the prompts whose
object it has learned to remove. Two readings, not yet separated: the LoRA has learned "when asked
for a chain saw, emit a still life", or the model freezes whenever it is prevented from rendering
what the prompt asks for. exp123's eta arms are the cheapest discriminator — a weaker erase pressure
that still erases should relax the freeze if it is a strength effect.

**Caveat on every number above:** `eval_num_prompts: 9` of 20, and exp102 showed the live monitor is
a prefix subset that is unbiased at base but blind after training. exp071 (all 200 prompts, all ten
classes) is the reported row.

## Downstream
- **exp071** — reported ESR/PSR on the final checkpoint (`frame_replace_lora_step600`, chosen because
  erasure is flat from step 200 and picking would be selection on the test set).
- **exp123** — `erase_esd_eta` ablation on the merged 33-row gen2 dataset, aimed at the freeze.

## Status
- [x] Datasets complete; config wired to exp117 + exp066 screened sets and exp068's anchors.
- [x] `merge_dataset.sh` run on the target cluster.
- [x] Submitted; completed 2026-08-16 (job 20735958, helios).
- [x] Checkpoint chosen for exp071 (step 600); results written up.
