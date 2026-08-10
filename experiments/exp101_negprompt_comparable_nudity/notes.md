---
status: done
concept: nudity
method: eval
thread: nudity
takeaway: >
  NegPrompt baseline on T2VUnlearning's own two CogVideoX-5B nudity sets (Gen 100, Ring-A-Bell 79),
  so our training-free row reads against theirs (46.35 / 14.91). exp083 ran NegPrompt but on I2P and
  our SafeSora filter, neither of which they evaluate for video. RESULT, and it changes the story:
  on Gen, NegPrompt is 0.39 frame / 0.34 video against a base of 0.414 / 0.360 — a ~6% relative
  reduction, i.e. it barely works. On Ring-A-Bell the same defence cuts 0.50 to 0.14. NegPrompt is
  strong on short art prompts (I2P, Ring-A-Bell) and weak on the long cinematic prompts of the set
  T2VUnlearning actually report, so the bar we must clear is set-dependent.
---
# exp101 — NegPrompt on the T2VUnlearning-comparable nudity sets

## Why
NegPrompt is the row a reviewer demands first ("why train anything — why not just negative-prompt
it?"), and it is one of the two training-free baselines T2VUnlearning reports. We already have a
strong NegPrompt result (exp083: I2P 0.105, SafeSora 0.240, both z≈3.7, and **no measurable quality
cost** — which overturned the expected "erases but wrecks generation" story and left our case resting
on the residual). But exp083's prompt sets are ours, and neither is a column T2VUnlearning reports
for video. This run puts NegPrompt on their sets.

Identical to exp083 in every field except the prompt sets — same negative prompt verbatim, same base
model, same unrelated control — so the two NegPrompt rows differ in exactly one thing.

## Setup
- run_001 → `prompts/cogvideox_nudity.csv` (100). **This is their released Gen set**, same prompts,
  same seeds — see [`docs/comparability_t2vunlearning.md`](../../docs/comparability_t2vunlearning.md).
- run_002 → `prompts/ring_a_bell_nudity.csv` (79), their redistributed Ring-A-Bell nudity prompts.

The negative prompt is applied to **every** set including `unrelated`: NegPrompt is a deployed
inference-time defence, so its collateral damage is part of what is being measured.

## What to watch
- **The bar is our own base row, not theirs.** exp063 measures Original at 41.4 (frame rate) on Gen
  where T2VUnlearning report 61.80. Until that 20-point discrepancy is resolved (their checkpoint
  through our eval, or a Q16 pass — see the doc), report reductions from *our* Original.
- **exp083's finding should reproduce or it needs explaining.** NegPrompt cost no measurable quality
  there. If it suddenly costs quality on these sets, one of the two results is an artefact.
- Report `nudity_frame_rate` (theirs) beside `nudity_detection_rate` (ours).

## Status
- [ ] Submitted (2 jobs, ~115 and ~94 clips).
- [ ] Scored on x86_64, same machine as every other reported row.
