---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
---
# exp153 — does rank 64's motion margin finally break the 0.15 floor at step 100, and does ESR/PSR
# keep improving there?

## Why
exp151 mapped rank 64 one save-interval earlier than exp149's step-300 checkpoint (step 200) and
found the ESR/PSR curve still rising (ESR-1 74.90, ESR-5 32.35, both better than step 300) while
the motion picture reversed: chain saw's own motion fell to 0.176, a margin of only 0.026 over the
0.15 guard floor (the thinnest of any checkpoint tested in this thread — exp149's step 300 had
0.378, exp148's step 600 had 0.181), and mean preserved-class motion loss rose to 49.8% — worse
than step 600's ~48% and far worse than step 300's ~17%. exp147's live 9-prompt monitor showed
concept top-1 already at 0.00 by step 100, one interval before step 200, so a checkpoint exists to
test whether ESR/PSR keeps climbing past step 200 and, more importantly, whether the erased-class
motion margin — already down to 0.026 — actually crosses under the 0.15 floor before that.

This directly probes the boundary of GOAL.md's motion guard, not just a preference between
checkpoints: if step 100 fails the guard, that caps how far "stop earlier" can be pushed for this
rank regardless of how good ESR/PSR look, and settles the question exp151 raised (is the motion
margin still eroding, or was step 200 near a local floor of its own).

## Hypothesis and what would falsify it
Hypothesis: step 100 continues the ESR/PSR improvement trend (ESR-1/ESR-5 at or above exp151's
74.90 / 32.35) but erased-class motion drops below the 0.15 guard floor, meaning step 200 was
already close to the last checkpoint that clears it — capping the "stop earlier" prescription for
rank 64 on this dataset regardless of ESR/PSR gains further back.

Falsified by:
- **Motion margin stays comfortably above 0.15** (e.g. flat or improved from step 200's 0.176) —
  would mean the margin erosion seen from step 300 to step 200 was not a monotonic trend either,
  matching the same non-monotonic shape the ESR/PSR-vs-motion story already showed once between
  step 300 and step 600.
- **ESR-1/ESR-5 clearly worse than step 200** — would mean step 100 is undertrained on the full
  protocol despite the live monitor's step-100 top-1 read, the same small-sample overstatement
  pattern seen before (exp135, exp139) but this time on the near side of the curve rather than the
  far side.

## Setup
Field-for-field exp151 (same 200-prompt protocol, same `erased_class: "chain saw"`, same 2B model,
`eval_inference_steps: 50`) except `lora_checkpoint_dir` points at exp147's **step-100** checkpoint
instead of step-200. No training job — exp147's checkpoints were saved every 100 steps and already
exist in the repo.

## What to watch
- **Restricted ESR-1/ESR-5/PSR-1/PSR-5** against exp151's step-200 row (74.90 / 32.35 / 80.15 /
  96.41) and exp149's step-300 row (74.49 / 21.63 / 79.97 / 91.87).
- **Erased-class (chain saw) motion against the 0.15 guard floor** — the central question this run
  answers. exp151's margin was 0.026; whether it survives one more save interval back.
- **Mean preserved-class motion loss vs exp130's per-class base** against exp151's 49.8% — does the
  collateral keep growing or has it peaked.

## Status
- [ ] Submitted.
- [ ] Row measured under both conventions; motion guard checked explicitly against the 0.15 floor;
      compared against exp151 (step 200) and exp149 (step 300) to find where the "stop earlier"
      prescription runs out for rank 64.
