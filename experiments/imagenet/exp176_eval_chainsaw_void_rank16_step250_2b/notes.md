---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run. Upper bracket (step 250) of exp173's rank-16 void-target LoRA — tests whether rank
  16's step-200 ESR-5 peak is narrow (collapses by step 250, like rank 32's did at exp169) or broad.
---
# exp176 — eval: chain-saw void-target dataset x rank 16, CogVideoX-2B, step 250

## Why
On the void-target dataset, rank 32's step-200 ESR-5 spike (exp163: 43.57) was narrow — exp169
(step 250) fell to 15.00, back at step-300's level, and exp166/exp167/exp168 confirmed no
checkpoint within ±50 steps reproduced it. Rank 8's peak was much broader (exp171 step 200 32.04,
exp170 step 300 30.20). This run reads which shape rank 16 has, and whether erased-class motion has
recovered by step 250 the way rank 32's did (exp169 motion 0.3645).

## Hypothesis and what would falsify it
Hypothesis: rank 16's peak is intermediate in width — step 250 keeps more of the peak than rank
32's did (which lost ~65% of its ESR-5 by this step) but less than rank 8's near-flat 200→300.

Falsified by: step 250 ESR-5 equal to or above exp175's step-200 read (peak not centred at 200 for
rank 16), or step 250 already at exp158's step-600 floor (15.41) with step 200 far above it (peak
as narrow as rank 32's).

## Setup
Eval-only, `job_type: eval`, `mode: imagenet`, exp173's pre-existing `frame_replace_lora_step250`,
identical 200-prompt protocol. Submitted alongside exp174 (150), exp175 (200), exp177 (300) —
independent, no dependency between them.

## Status
- [ ] Submitted.
- [ ] Compared against exp174/exp175/exp177 and the rank-8 / rank-32 void curves.
