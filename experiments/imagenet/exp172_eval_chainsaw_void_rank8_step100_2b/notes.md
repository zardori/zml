---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  PARTIAL FALSIFICATION: step 100 does not continue rising, it sits below both step 200 and step
  300. Restricted (10-way) row: ESR-1 64.80, ESR-5 23.98, PSR-1 86.54, PSR-5 96.19, motion (chain
  saw) 0.7977 -- the healthiest motion margin of the four checkpoints. Against exp171 (step 200:
  68.88/32.04) and exp170 (step 300: 69.69/30.20), ESR-1 and ESR-5 are both lower here, so the
  void+rank-8 curve across 100/200/300/600 is a single peak at step 200 (23.98 -> 32.04 -> 30.20 ->
  15.41), not a monotonic climb to the earliest checkpoint the way rank 64's random-distractor curve
  was (exp149->exp151->exp153) -- it matches rank 32's own non-monotonic single-peak shape
  (exp150 vs exp152/exp154), just far milder in amplitude and, unlike rank 32's version, never
  breaching the motion guard at any of the four checkpoints. PSR-1/PSR-5 are this LoRA's best of the
  four checkpoints (86.54/96.19), continuing the pattern seen throughout this thread where weaker
  erasure reads as stronger preservation -- the mirror image of collateral damage, not an
  independent gain. This closes the void+rank-8 checkpoint sweep: step 200 (exp171) is the operating
  point on this LoRA, at ESR-1 68.88 / ESR-5 32.04, comfortably legal on both PSR floors and the
  motion guard, but well short of exp153's thread-best legal row (rank 64/step 100:
  ESR-1 77.86 / ESR-5 44.49).
submitted: 2026-08-27 21:18 helios job 21367441
---
# exp172 — eval: chain-saw void-target dataset x rank 8, CogVideoX-2B, step 100

## Why
See exp170 for the full rationale. This is the earliest checkpoint exp157 saved. exp157's live
9-prompt monitor showed concept top-1 already at 0.00 by step 100 (faster than exp133's
identical-recipe/random-distractor run, which took until step 200), so step 100 is not obviously
undertrained despite being the earliest available checkpoint — the same reasoning that made rank
64's step-100 checkpoint (exp153) worth evaluating and it turned out to be that thread's best
full-protocol row.

## Hypothesis and what would falsify it
Hypothesis: step 100 continues any ESR-5 gain seen at exp170 (step 300) / exp171 (step 200) rather
than reversing it — mirroring rank 64's monotonic climb all the way to step 100 (exp153), not rank
32's single non-monotonic peak (exp150 vs exp152/exp154).

Falsified by: ESR-5 dropping back toward or below exp158's step-600 read (15.41) after rising at
step 300/200, indicating a peak-and-reverse shape instead of a monotonic one — or by ESR-5 never
having risen at exp170/exp171 in the first place, in which case this checkpoint has nothing left
to falsify beyond confirming the flat/null result.

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp157's `frame_replace_lora_step100`
(pre-existing checkpoint, no new training), identical 200-prompt protocol. Submitted alongside
exp170 (step 300) and exp171 (step 200) — independent evals, no dependency between them.

## Status
- [x] Submitted.
- [x] Compared against exp158 (step 600), exp170 (step 300), exp171 (step 200). See frontmatter.
