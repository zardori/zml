---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  NOT FALSIFIED (STEP 300 > STEP 600 ON THIS LORA TOO), BUT STILL FAR BELOW EXP150'S PEAK — THE
  INTERFERENCE EXP161 FOUND IS NOT AN ARTEFACT OF OVERTRAINING. Restricted (10-way) row: ESR-1
  60.41, ESR-5 18.47, PSR-1 86.96, PSR-5 95.99. Against exp161 (same LoRA, step 600: 45.71 / 10.82
  / 90.25 / 97.07): ESR-1 up 14.70, ESR-5 up 7.65, both PSR cells down slightly — the direction
  exp150 found for rank 32 on the random-distractor dataset (earlier stop helps ESR/PSR-5 at a
  small PSR-1 cost) reproduces here too, so per this run's pre-registered falsifier (scoring at or
  below exp161 would falsify) the early-stop effect generalizes across datasets, not just across
  ranks. But the magnitude does not: exp150 (rank 32, random-distractor, step 300: 72.76 / 38.67 /
  84.46 / 96.49) beats this row on ESR-1 (+12.35) and ESR-5 (+20.20 — more than double) while
  losing only on PSR-1 (-2.50). So stopping earlier recovers some of what void-target cost this
  combination, but nowhere near enough to close the gap with rank-32-alone's own early-stop
  optimum — the interference exp161 flagged is a real property of combining these two levers, not
  just a step-600 overtraining artefact that early stopping fixes. Chain saw's own restricted
  top-5 and quality block not separately called out here; the headline is the ESR-1/ESR-5 gap to
  exp150. exp163 and exp164 (step 200 and step 100 evals of this same LoRA, queued this tick) test
  whether the decline-with-training-length trend continues further back, the way rank 64's
  ESR/PSR curve kept rising all the way to step 100 (exp149→exp151→exp153).
---
# exp162 — eval: chain-saw void-target dataset x rank 32, CogVideoX-2B, early-stop checkpoint (step 300)

## Why
exp150 found that rank 32's step-300 checkpoint (random-distractor dataset, half the training
budget) beats its own step-600 read (exp143) on ESR-1, ESR-5 (nearly doubling it, 20.92 → 38.67 —
this thread's best ESR-5 to date) and PSR-5, at a small PSR-1 cost. exp160 (void-target dataset,
rank 32) saved the same per-100-step checkpoints as exp142/exp147, so this is an eval-only
diagnostic against exp160's own step-300 checkpoint, paired with exp161's step-600 read — no new
training required.

Live monitor at step 300 (from exp160's run): concept top-1 0.00, top-5 0.00, motion 0.2017 —
above the 0.15 guard floor with a similar margin to exp142's own step-300 live read, which is the
checkpoint exp150 went on to confirm as a genuine local optimum for rank 32.

## Hypothesis and what would falsify it
Hypothesis: the step-300 early-stop effect exp150 found for rank 32 on the random-distractor
dataset generalizes to the void-target dataset — i.e. this checkpoint's ESR-5 is higher than
exp161's step-600 read on the same LoRA, mirroring exp150 vs exp143's step-600/step-300 gap.

Falsified by: this checkpoint scoring at or below exp161's step-600 read on ESR-1 and ESR-5 (the
exp152 outcome — step 200 was uniformly worse than step 300 for the random-distractor rank-32 run,
showing "earlier is always better" is not a safe assumption).

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp160's `frame_replace_lora_step300` checkpoint
(half the 600-step training budget), identical 200-prompt protocol to every other row in this
thread.

## Status
- [ ] Submitted.
- [ ] Compared against exp150 (rank 32, random-distractor, step 300) and exp161 (same LoRA,
      step 600) cell-for-cell.
