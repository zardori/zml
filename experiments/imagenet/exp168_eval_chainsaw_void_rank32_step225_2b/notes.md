---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  FIRST FALSIFIER FIRED, SECOND DID NOT — step 225 collapses toward step 300's level while motion
  recovers. Restricted (10-way) row: ESR-1 65.31, ESR-5 19.69, PSR-1 86.94, PSR-5 96.28. Against
  exp163's step-200 spike on the same void+rank-32 LoRA (86.73 / 43.57 / 82.95 / 93.45): ESR-1 down
  21.42, ESR-5 down 23.88 — landed right on exp162's step-300 read (60.41 / 18.47), not "step 200
  minus a bit", so the classification peak is narrower than ±25 steps, exactly as exp166 (step 150,
  72.65 / 18.37) and exp167 (step 175, 59.59 / 18.37) found from below and exp169 (step 250,
  66.94 / 15.00) from above. Erased-class motion is 0.389, well clear of the 0.15 floor (step 200
  was 0.1379, the only breach) — the motion recovers on the peak's trailing edge just as exp162's
  step-300 read (0.408) predicted, so the freeze at step 200 is a sharp isolated dip coincident
  with the ESR spike, not a plateau. Preserved-class mean motion loss vs exp130's per-class base is
  ~35% (cassette player -81%, French horn -59%, gas pump -52% worst; tench/golf ball essentially
  unaffected), in the same band as every other void+rank-32 checkpoint. Net: this closes the
  exp165–exp169 fine-grained bracket — no checkpoint within ±50 steps of exp163's step-200 spike
  reproduces more than a fraction of its erasure, and only step 200 itself breaches the motion
  floor. exp150's rank-32/step-300 random-distractor row (72.76 / 38.67, motion 0.296) remains a
  cleaner and better full-protocol row than anything on this void+rank-32 LoRA.
---
# exp168 — eval: chain-saw void-target dataset x rank 32, CogVideoX-2B, step 225

## Why
Immediate upper neighbour of exp163's step-200 classification peak (thread-best restricted ESR-1
86.73 / ESR-5 43.57, erased-class motion 0.1379 < 0.15 floor). The trailing edge is where motion
recovers with more training (exp162's step-300 motion was 0.408), so it is the most likely place
for a checkpoint that keeps the erasure and clears the floor. See exp166 for the full rationale;
this run evaluates exp165's step-225 checkpoint.

## Hypothesis and what would falsify it
Hypothesis: step 225 retains most of step 200's erasure (ESR-1 70s–80s, ESR-5 30s–40s) while
clearing the 0.15 motion floor — a usable operating point on the peak's recovering edge.

Falsified by: ESR-1/ESR-5 already collapsed toward exp162's step-300 read (60.41/18.47) — the peak
is narrower than ±25 steps — OR erased-class motion_score_mean still below 0.15 at step 225.

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp165's `frame_replace_lora_step225`, identical
200-prompt protocol. Submitted alongside exp166 (150), exp167 (175), exp169 (250).

## Status
- [x] Submitted.
- [x] Compared against exp163 (step 200), exp162 (step 300), and the other brackets. See frontmatter.
