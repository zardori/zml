---
status: active
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  NegPrompt baseline row, to separate 'our erasure works' from 'our protocol is easy to score
  well on'.
---
# exp065 — NegPrompt baseline on the ImageNet object protocol

## Goal
Fill the `NegPrompt` comparison row for the two pilot classes. T2VUnlearning reports NegPrompt at
ESR-1 48.59 / ESR-5 19.79 / PSR-1 65.37 / PSR-5 88.62 — a method that distorts the target enough to
confuse a classifier without actually removing it, which is the behaviour frame_replace has to beat.

Having it in *our* setup matters more than the published number: it separates "our erasure works"
from "our protocol is easy to score well on".

## Setup
Identical to exp064 except `erased_class` is set and `negative_prompt: auto` (which resolves to the
class name at generation time). Still the unmodified base model — no LoRA, no training.

`erased_class: ["chain saw", "church"]` is list-valued, so `submit_job.py` grids it into two runs
under `grid_{TIMESTAMP}/run_001` and `run_002`.

`./submit_job.py athena experiments/exp065_negprompt_imagenet/config.yaml`

## What to watch
- ESR-1 vs. exp064's ESR-1 for the same class: the lift is what negative prompting alone buys.
- PSR relative to exp064: negative prompting is global, so some preservation loss is expected.
- ESR-5 vs. ESR-1: the paper's argument is that baselines move ESR-1 but not ESR-5, because the
  object is distorted rather than removed. Worth checking whether that holds here too.

## Status
- [ ] Submitted (independent of everything else; can run any time after exp064).
- [ ] Results pulled, NegPrompt row recorded.
