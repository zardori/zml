---
status: abandoned
concept: fire
method: eval
thread: baselines
takeaway: >
  Planned base-model baseline on the full exp047 prompt sets. Never ran (noted as missing in
  exp053); exp055 measured the base model instead.
---
# exp052 — base-model baseline on the full exp047 prompt sets

## Hypothesis
Two gaps make current results hard to read:

1. **The hard related_v2 set has no baseline.** exp047 (exp046@500 checkpoint) scored
   `fire_detection_rate` 0.3 on related_v2 — but those prompts are deliberately fire-*colored*
   fire-free scenes (autumn leaves, orange silk, marigolds…), so 0.3 could be detector false
   positives rather than the LoRA injecting fire. Without the base model's rate on the same
   `(prompt, seed)` pairs the number is uninterpretable.
2. **No single base-model reference exists for the full sets.** Every unlearning run's
   concept/unrelated scores are compared against assumptions (concept detection "should be"
   ~1.0, unrelated clip ~0.33) rather than measured values on the exact committed prompt+seed
   pairs.

This run evaluates the unmodified CogVideoX-5b on all three full sets used in exp047
(15 fire + 10 related_v2 + 15 unrelated), saving all 40 videos and computing all scores.

## Pipeline
No code changes — `zml/eval/eval_model.py` already runs the base model when
`lora_checkpoint_dir` is unset, saves every video under `eval_step_0/<set>/` and writes
`metrics.json` (as in exp047's outputs).

`./submit_job.py athena experiments/archive/baselines/exp052_eval_base_full_sets/config.yaml`

Same job shape as exp047 (40 videos @ 50 steps), which fit in 6 h → `slurm_time: "0-6:00:00"`.
`scripts/eval.py` now writes `runtime.json` into the output dir, so the measured duration is
available for future eval budgeting.

## What to watch
- **related_v2 detection (the calibration):** baseline ≈ 0.3 → exp047's related score is
  detector noise and related_v2 preservation was clean; baseline ≈ 0 → the exp046 LoRA
  actually pushes fire-like content into fire-free scenes, a real bleed problem.
- **Concept baseline:** detection expected near 1.0 with healthy clip/colorfulness — the
  denominator for all future erasure claims (a run's "detection 0.2" only means something
  relative to this).
- **Unrelated baseline:** clip and colorfulness levels on the exact committed seeds, replacing
  the assumed clip ≈ 0.33 bar.
- Videos worth eyeballing: the related_v2 clips the detector flags, to see *what* it fires on.

## Results
- (pending run)
