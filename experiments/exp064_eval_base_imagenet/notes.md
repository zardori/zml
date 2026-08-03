---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Gate passed: chain saw (.513 top-1) and church (.733) both render, so the pilot is viable. Base
  mean top-1 is 55% vs the paper's 78% purely because of ImageNet sibling classes (cassette
  player/cassette/tape player, springer/setter, tench/barracouta) — restricting the ranking to the
  ten protocol classes gives 90% and reproduces their top-5 row, so we now report both conventions.
  frame_concept_threshold calibrated to 0.05 (chain saw) / 0.03 (church).
---
# exp064 — base-model ESR/PSR on the ImageNet object protocol

## Goal
Establish the `Original` row of the object-erasure comparison, and — before any GPU time goes into
datasets or training — check that our instrumentation reproduces roughly what the literature sees on
an unmodified model. Everything downstream (exp066–exp071) is read relative to this run.

Protocol, classifier and metric definitions: `docs/imagenet_objects.md`.

## Setup
`job_type: eval`, `mode: imagenet`, no `lora_checkpoint_dir`, no `erased_class`. All 200 prompts
(10 classes x 20) from `prompts/imagenet_objects.csv`, 49 frames, 50 inference steps, per-row seeds
from the CSV. Videos land in `eval_step_0/<class_slug>/video_{i}.mp4`; every frame is classified by
ResNet-50 (`IMAGENET1K_V2`) and the report is written to `esr_psr.json`.

With `erased_class` unset the report holds `per_erased_class` (ESR/PSR for each of the ten choices)
plus `mean`/`std` — which is exactly how the papers' `Original ±` numbers arise.

`./submit_job.py athena experiments/exp064_eval_base_imagenet/config.yaml`

## What to watch
- **Sanity gate.** T2VUnlearning reports `Original` ESR-1 21.62±20.13 / ESR-5 5.09±8.23 /
  PSR-1 78.38±2.24 / PSR-5 94.91±0.92 on CogVideoX-2B. We are on 5b with 49-frame clips, so exact
  agreement is not expected — but ESR-1 should be low (tens, not eighties) and PSR-1 high. An ESR-1
  near 80 means prompts that do not render their class, or a classifier/preprocessing bug. Fix that
  before spending on exp066/067.
- **Per-class top-1/top-5** in `esr_psr.json` `per_class`. A class the base model barely renders
  (top-1 near 0) is untrustworthy as an erasure target — its ESR starts at ceiling and cannot move.
  Note any such class here; it may need better prompts before it joins the pilot.
- **Threshold calibration.** Pull the `chain_saw/` and `church/` videos and run
  `uv run python -m zml.benchmarks.check_for_object --input_dir <dir> --target_class "chain saw"`
  locally to pick `frame_concept_threshold` for exp066/exp067. Reference point: a clean stock photo
  of a golf ball scores top-1 with probability ~0.44, so the per-frame probability of a genuinely
  present object sits well below 1.0 and the threshold must be set accordingly.
- `quality` block (clip/colorfulness/motion per class) — the base levels exp071 must hold.

## Results (`outputs_20260802_204935`)

Ran on athena in **5.71 h** for 200 videos, comfortably inside the 10 h allotment. (The earlier
`*_20260802_204313` pair is an aborted submission that produced nothing.)

### The `Original` row — mean ± std over the ten choices of erased class

| convention | ESR-1↑ | ESR-5↑ | PSR-1↑ | PSR-5↑ |
|---|---|---|---|---|
| ours, 1000-way | 45.01 ± 25.22 | 23.90 ± 20.11 | 54.99 ± 2.80 | 76.10 ± 2.23 |
| ours, 10-way restricted | 9.91 ± 9.57 | 3.44 ± 5.56 | 90.09 ± 1.06 | 96.56 ± 0.62 |
| T2VUnlearning (2B, 17f) | 21.62 ± 20.13 | 5.09 ± 8.23 | 78.38 ± 2.24 | 94.91 ± 0.92 |

Note that mean ESR-1 and mean PSR-1 sum to 100 by construction — averaged over all ten choices of
erased class, both reduce to the overall mean top-1 accuracy. The row carries two numbers, not four.

**Classification is not bit-reproducible across GPUs.** The job's own `esr_psr.json` (scored on the
A100 alongside generation) read 44.99 / 23.48 / 55.01 / 76.52; re-scoring the identical video files
locally to add the restricted block gave the table above. Per-class top-1 moves by ~0.002 and ESR-5
by 0.42 — different cuDNN kernels flipping frames whose top-1/top-2 margin is tiny. The refactored
scoring path was verified numerically identical to the pre-refactor expressions on a fixed `probs`
tensor, so this is hardware, not code. **Treat sub-1-point differences between runs as noise.**

### Gate: passed

**Both pilot classes render well.** chain saw top-1 .513 / top-5 .791, church .733 / .950 — so ESR
has real headroom on both (base ESR-1 48.7 and 26.7). Per-class base top-1:

| class | 1000-way top-1/top-5 | 10-way top-1/top-5 |
|---|---|---|
| parachute | .791 / .852 | .935 / 1.000 |
| garbage truck | .781 / .997 | 1.000 / 1.000 |
| golf ball | .755 / .784 | .884 / 1.000 |
| gas pump | .736 / .844 | .847 / .943 |
| church | .733 / .950 | 1.000 / 1.000 |
| French horn | .589 / .852 | .898 / .900 |
| chain saw | .513 / .791 | .946 / .993 |
| tench | .369 / .702 | .915 / .992 |
| English springer | .158 / .259 | .648 / .829 |
| cassette player | .075 / .580 | .937 / 1.000 |

**The gap to the paper is ImageNet's taxonomy, not our pipeline.** Top-1 confusion on the three weak
classes: `cassette player` → *cassette* 21.9% + *tape player* 18.9% (only 7.4% on the class itself —
the object is rendered correctly, ResNet-50 is splitting three near-duplicates); `English springer` →
*Gordon setter* 11.6%, *English setter* 5.4%; `tench` → *barracouta* 21.0%, *gar* 6.6%. Better
prompts cannot fix this, which is what motivated reporting the restricted convention alongside —
see `docs/imagenet_objects.md` §3.1. If the pilot ever extends to all ten classes, those three are
unreliable 1000-way erasure targets: their ESR starts near the ceiling and has nowhere to move.

### Threshold calibration

Scoring each pilot class against the other nine classes' clips (8820 negative frames vs 980 positive):

| class | negative p99.9 / max | positive p25 / p50 / p75 | chosen | TPR | FPR |
|---|---|---|---|---|---|
| chain saw | 0.018 / 0.044 | 0.024 / 0.115 / 0.385 | **0.05** | 64.7% | 0.0% |
| church | 0.003 / 0.006 | 0.127 / 0.233 / 0.313 | **0.03** | ≥85% | 0.0% |

Separation is wide enough that false positives cost nothing in this range, so the shipped 0.15 guess
was strictly worse than 0.05 — no precision gained, a fifth of the real concept frames discarded.
exp066/exp067 updated accordingly.

### Carried forward into exp066/exp069

- **exp066 yield risk.** 5 of 20 full-concept chain-saw clips have *no* frame above 0.05, and a split
  clip devotes only part of its length to the concept — expect real `no_concept` attrition on the 30
  triples. If fewer than ~12 survive review, extend `prompts/split_imagenet_chain_saw.csv` rather
  than dropping the threshold below the measured negative ceiling.
- **exp069 live signal.** Base `object_detection_rate` (clips with ≥50% of frames top-1) is 10/20 for
  chain saw and 15/20 for church. Chain saw therefore has only half the dynamic range during
  training — read it as directional and treat exp071's frame-pooled ESR as the verdict.

## Status
- [x] Submitted.
- [x] Results pulled; `Original` row recorded; per-class weak spots noted.
- [x] `frame_concept_threshold` calibrated for chain saw and church.
