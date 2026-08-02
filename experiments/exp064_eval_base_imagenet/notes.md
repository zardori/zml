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

## Status
- [ ] Submitted.
- [ ] Results pulled; `Original` row recorded; per-class weak spots noted.
- [ ] `frame_concept_threshold` calibrated for chain saw and church.
