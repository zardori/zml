---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  NOT FALSIFIED, AND THIS IS THE THREAD'S BEST CLASSIFICATION ROW -- BUT IT FAILS THE MOTION GUARD.
  Restricted (10-way) row: ESR-1 86.73, ESR-5 43.57, PSR-1 82.95, PSR-5 93.45. Against exp162
  (same LoRA, step 300: 60.41 / 18.47 / 86.96 / 95.99): ESR-1 up 26.32, ESR-5 up 25.10 -- both far
  above the falsifier bar (scoring at or below exp162), so the decline-with-training-length trend
  exp161->exp162 reported does NOT continue monotonically; it reverses hard one checkpoint
  earlier. ESR-1 86.73 is within 5.65 points of GOAL.md's target (92.38) -- closer than anything
  else in the thread -- and ESR-5 43.57 essentially ties this thread's previous best (exp153,
  rank 64/step 100: 44.49). BUT the erased-class motion guard FAILS here: chain saw
  `motion_score_mean` is 0.1379, below GOAL.md's 0.15 floor (exp130 base: 0.840) -- the first time
  in this thread's entire rank/step sweep that a fully-evaluated checkpoint has actually breached
  the floor rather than just approaching it (exp151's rank64/step200 came closest before this, at
  0.176). Motion at the three other checkpoints on this same LoRA (step 100: 0.820, step 300:
  0.408, step 600: 0.546) all clear the floor comfortably, so this is a sharp, narrow dip
  coincident with the ESR spike, not a trend. PSR-1/PSR-5 both stay well clear of their floors
  (54.03 / 82.14). Net: step 200 is a genuine local optimum for classification erasure on this
  LoRA (mirroring rank-32-alone's own non-monotonic peak at step 300, exp150, just shifted
  earlier) but it cannot be reported as a target-clearing checkpoint because of the motion guard --
  the strongest erasure and the motion floor breach land on the exact same checkpoint, which is
  the mechanistic pattern this whole thread has been flagging (freeze co-occurring with strong
  suppression) rather than a coincidence. exp165 maps the curve at finer resolution around this
  peak (steps 125-275, interval 25) to see whether a nearby checkpoint keeps most of the ESR gain
  while staying clear of the motion floor.
---
# exp163 — eval: chain-saw void-target dataset x rank 32, CogVideoX-2B, step 200

## Why
exp161 (step 600) and exp162 (step 300) both found that stacking void-target data (exp156) with
rank-32 capacity underperforms EITHER lever alone on ESR-1/ESR-5, even though the two levers each
independently moved a different half of GOAL.md's target (void → ESR-1/PSR, rank32+early-stop →
ESR-5). Step 300 partially recovered from step 600's read (ESR-1 45.71→60.41, ESR-5 10.82→18.47)
but still falls well short of exp150's rank-32-alone step-300 peak (ESR-1 72.76, ESR-5 38.67). This
is an eval-only diagnostic against exp160's already-trained step-200 checkpoint — no new training
required — to see whether the decline-with-training-length trend continues, the way rank 64's
ESR/PSR kept climbing all the way to step 100 (exp149→exp151→exp153) rather than peaking at step
300 the way rank 32 alone did (exp150's peak, exp152's step-200 drop below it).

## Hypothesis and what would falsify it
Hypothesis: step 200 continues the trend from step 600→300 (ESR-1/ESR-5 both higher again),
possibly approaching exp150's rank-32-alone numbers — i.e. void+rank32's interference is a
training-length effect that early stopping can substantially undo, not a hard ceiling.

Falsified by: this checkpoint scoring at or below exp162's step-300 read on ESR-1 or ESR-5 — the
exp152 outcome (step 200 was uniformly worse than step 300 for rank-32-alone), which would mean
step 300 is a local peak for this combination too, mirroring rank 32 alone rather than rank 64.

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp160's `frame_replace_lora_step200` checkpoint,
identical 200-prompt protocol to every other row in this thread. Submitted alongside exp164
(step 100) — independent evals of the same completed training run's saved checkpoints, no
dependency between them.

## Status
- [ ] Submitted.
- [ ] Compared against exp162 (same LoRA, step 300), exp150 (rank 32, random-distractor, step 300)
      and exp164 (same LoRA, step 100).
