---
status: done
concept: nudity
method: eval
thread: nudity
takeaway: >
  The "Original" row, and far more load-bearing than it looked. Its prompt set turns out to BE
  T2VUnlearning's released Gen set (same 100 prompts and seeds), so this is our base row on their
  exact column: nudity_frame_rate 0.414, detection_rate 0.360, motion 0.686. The run died in its
  CPU scoring phase on 2026-08-02 and sat unscored for a week; recovered from the clips alone on
  2026-08-10 with tools/score_eval_videos.py. Also carries DOVER and the Q16 `unsafe` rate (0.516).
---
# exp063 — base-model baseline on exp062's nudity eval sets

## Goal
exp062 (frame_replace, nudity, eta=2) reports `nudity_detection_rate` / clip / colorfulness /
motion at each checkpoint, but there is no unmodified-base-model reference on the exact same
sets (`prompts/cogvideox_nudity.csv` concept, `prompts/cogvideox_fire_control_unrelated.csv`
unrelated) to read those numbers against — same gap exp052 closed for the fire concept.
Without this, exp062's "detection dropped to 0.4 by step 600" is uninterpretable: is base
already <1.0 on these particular (prompt, seed) pairs? What's the base unrelated clip/motion
level these runs should hold?

## Setup
Same pattern as exp052: `job_type: eval`, no `lora_checkpoint_dir` (evaluates raw
CogVideoX-5b), full sets (no `eval_num_prompts` cap) so every row is generated and scored —
100 concept + 16 unrelated = 116 videos, all saved under `eval_step_0/<set>/`. No related set
(none exists for nudity yet — exp062 doesn't score one either).

`./submit_job.py athena experiments/nudity/exp063_eval_base_nudity/config.yaml`

exp052's 40 videos @ 50 steps took 1.2h wall-clock (see its `runtime.json`); 116 videos here
scales to ~3.5h, so `slurm_time: "0-06:00:00"` leaves headroom.

## What to watch
- **Concept baseline (`nudity_detection_rate` on `cogvideox_nudity.csv`):** expected near
  1.0 — the denominator exp062's erasure numbers are relative to.
- **Unrelated baseline:** clip/colorfulness/motion levels on the exact committed seeds,
  replacing assumed reference values.
- Once this lands, fold both into exp062's notes as the "vs. base" comparison row.

## Status
- [ ] Submitted.
- [ ] Results pulled, exp062 comparison written up.
