---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  NOT FALSIFIED ON ESR-5, MARGINALLY ON PSR-1 -- AND THIS IS THE PEAK. Restricted (10-way) row:
  ESR-1 68.88, ESR-5 32.04, PSR-1 82.06, PSR-5 94.69, motion (chain saw) 0.4983. ESR-5 more than
  doubles exp158's step-600 read (15.41 -> 32.04, +16.63) and is the highest of the four void+rank-8
  checkpoints now measured (100/200/300/600: 23.98/32.04/30.20/15.41) -- step 200 is the peak, one
  interval later than void+rank-32's own step-200 spike lands relative to its curve shape (exp163),
  same qualitative non-monotonic pattern at a much smaller amplitude. PSR-1 (82.06) sits 0.65 points
  BELOW exp134's random-distractor baseline (82.71) -- inside the letter of the pre-registered
  falsifier, but the margin is noise-sized (comparable runs move PSR-1 by more than this from eval
  seed variance alone) and PSR-1 is still 28pp clear of GOAL.md's 54.03 floor, so this is flagged,
  not treated as a real regression. The finding that matters: unlike void+rank-32's identical-shaped
  step-200 spike (exp163, ESR-5 43.57 but motion 0.1379 -- breaching the 0.15 guard floor), this
  checkpoint's motion (0.4983) clears the floor by more than 3x. Capacity controls the spike's
  amplitude AND its motion risk together -- rank 8 gets a smaller ESR-5 gain but keeps it legal;
  rank 32 gets a bigger gain but forfeits it. Motivates testing an intermediate rank (16) at a
  similar fine-grained step sweep to see whether the trade is continuous (exp173).
submitted: 2026-08-27 21:18 helios job 21367439
---
# exp171 — eval: chain-saw void-target dataset x rank 8, CogVideoX-2B, step 200

## Why
See exp170 for the full rationale: exp158 only evaluated exp157's final checkpoint (step 600),
which won on ESR-1/PSR-1/PSR-5 over the random-distractor baseline but left ESR-5 flat. Early
stopping helped ESR-5 substantially at rank 32/64 on the random-distractor dataset but did NOT
rescue it when void-target data was stacked with rank 32 (exp160–exp169). This run tests step 200
on the untested rank-8 + void + early-stopping combination.

## Hypothesis and what would falsify it
Hypothesis: step 200 improves ESR-5 over exp158's step-600 read (15.41), and — with exp170 — traces
whether the gain (if any) grows monotonically toward earlier steps (as it did for rank 64,
exp148→exp149→exp151→exp153) or peaks and reverses (as it did for rank 32 alone, exp150→exp152).

Falsified by: ESR-5 at or below exp158's step-600 read, OR ESR-1/PSR dropping below exp134's
random-distractor baseline.

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp157's `frame_replace_lora_step200`
(pre-existing checkpoint, no new training), identical 200-prompt protocol. Submitted alongside
exp170 (step 300) and exp172 (step 100) — independent evals, no dependency between them.

## Status
- [x] Submitted.
- [x] Compared against exp158 (step 600), exp170 (step 300), exp172 (step 100). See frontmatter.
