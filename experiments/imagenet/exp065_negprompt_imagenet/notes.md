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

`./submit_job.py athena experiments/imagenet/exp065_negprompt_imagenet/config.yaml`

## What to watch
- ESR-1 vs. exp064's ESR-1 for the same class: the lift is what negative prompting alone buys.
- PSR relative to exp064: negative prompting is global, so some preservation loss is expected.
- ESR-5 vs. ESR-1: the paper's argument is that baselines move ESR-1 but not ESR-5, because the
  object is distorted rather than removed. Worth checking whether that holds here too.

## Run 1 (`grid_20260803_233332`, athena) — timed out, resubmit

Both arms hit the 10 h wall clock partway through `golf_ball` and wrote **no `esr_psr.json`**:
run_001 (chain saw) 163/200 videos, run_002 (church) 164/200. Nothing is wrong with the results that
did land — generation is resumable, so a resubmission picks up the remaining ~37 per arm.

Measured from video mtimes, both arms ran at **219 s/video**, so a cold 200-video run needs ~12.2 h.
`slurm_time` raised 10 h → **14 h**, sized for a cold start rather than for the resume.

That rate is **2.1x exp064's** (200 videos in 5.71 h = 103 s/video) on the same cluster, same 50
inference steps, same 200 prompts — and a negative prompt adds no forward pass, so the cause is not
the thing being tested. Most likely node-to-node variation or contention between the two concurrent
grid arms, but the logs record neither node nor elapsed time, so it cannot be settled after the fact.
This run is why `slurm/run_info.sh` now exists: every job writes `$OUTPUT_DIR/run_info.json` with
cluster, node, elapsed and outcome, **including on timeout**, and `tools/experiments_index.py`
surfaces it as a column in `INDEX.md`.

## Status
- [x] Submitted (independent of everything else; can run any time after exp064).
- [x] Run 1 timed out at 10 h — 163/200 and 164/200, no report written.
- [ ] Resubmitted at 14 h (resumes from the existing videos).
- [ ] Results pulled, NegPrompt row recorded.
