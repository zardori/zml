---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  DONE (the resubmission landed 2026-08-08; the report sat unread until 2026-08-16). And it splits
  hard by ranking convention, which is the finding. 1000-way, NegPrompt looks strong — chain saw
  ESR-1 49.4 -> 70.8, church 26.1 -> 75.1. Restricted to the ten protocol classes it erases almost
  nothing: chain saw ESR-1 17.1 / ESR-5 0.00, church ESR-1 0.2 / ESR-5 0.00. So most of its apparent
  erasure is sibling-class confusion, not removal — exactly the "distorts without removing" behaviour
  T2VUnlearning attributes to it, and the reason our row must be read 10-way. Bar for exp071.
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

## Results (`grid_20260808_235029`, both arms complete, 200 videos each)

| | ESR-1 | ESR-5 | PSR-1 | PSR-5 | DOVER tech |
|---|---|---|---|---|---|
| **chain saw**, 1000-way | 70.92 | 44.39 | 53.07 | 71.58 | 0.094 |
| chain saw, restricted | **17.24** | **0.00** | 83.36 | 93.55 | |
| base (exp064), 1000-way | 48.67 | 20.92 | 55.40 | 75.77 | 0.100 |
| base, restricted | 5.41 | 0.71 | 89.59 | 96.26 | |
| **church**, 1000-way | 75.10 | 29.90 | 53.00 | 69.43 | 0.085 |
| church, restricted | **0.20** | **0.00** | 87.26 | 93.82 | |
| base (exp064), 1000-way | 26.73 | 5.00 | 52.96 | 74.00 | 0.101 |
| base, restricted | 0.00 | 0.00 | 88.99 | 96.18 | |

**The two conventions disagree about whether NegPrompt works at all**, and that is the result worth
carrying. Ranked over ImageNet-1k it looks like a serious defence (church ESR-1 26.7 → 75.1). Ranked
within the ten protocol classes it erases essentially nothing: church ESR-1 goes 0.00 → 0.20, and
ESR-5 is 0.00 for both classes, i.e. **the true class is still in the model's top five in every
video**. The 1000-way gain is the object surviving while the classifier's *first* choice slides to a
sibling — precisely the "distorts the target enough to confuse a classifier without removing it"
behaviour T2VUnlearning ascribes to NegPrompt, now measured in our own setup.

Consequence for the thread: **report the restricted column, or report both.** A frame_replace row
quoted 1000-way against NegPrompt's 70.8 is arguing with an artefact.

Quality is untouched (chain saw motion 1.11 and clip 0.302 against base 0.563/0.322 — motion is
*higher*; DOVER technical 0.094 / 0.085 against base 0.100 / 0.101), so there is no degeneration to
point at either. This is the row frame_replace has to beat **without** the freeze exp069 shows.

## Status
- [x] Submitted (independent of everything else; can run any time after exp064).
- [x] Run 1 timed out at 10 h — 163/200 and 164/200, no report written.
- [x] Resubmitted at 14 h (resumed from the existing videos); both arms completed 2026-08-08.
- [x] Results pulled, NegPrompt row recorded — see the table above.
- [x] Re-scored locally 2026-08-16 to add DOVER (`imagenet_eval --rescore`, provenance flags passed
      so the report keeps its `negative_prompt`).
