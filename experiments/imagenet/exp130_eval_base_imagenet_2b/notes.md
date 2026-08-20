---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  GATE PASSED, and it passed better than 5b did. Restricted (10-way) `Original` row: ESR-1 10.60,
  ESR-5 2.09, PSR-1 89.40, PSR-5 97.91 — all four cells beat T2VUnlearning's published Original
  (21.62/5.09/78.38/94.91) in the direction that matters for a baseline (lower ESR, higher PSR), and
  no class is degenerate: restricted top-1 ranges 0.52 (English springer, the one weak spot) to 1.00
  (church). Per-class flips the 5b ordering — church renders at restricted top-1 1.00, chain saw at
  0.88, so "church is the hard scene-level class" (exp070/exp128's 5b finding) is not established at
  2B from this run alone. 1000-way per-class top1 (chain saw 0.369, church 0.549) is lower than 5b's
  (0.513/0.733), so 2B is weaker on fine-grained taxonomy but at least as strong on rendering the
  object recognizably. Unblocks 2B dataset/training spend; exp131 is the first one.
---
# exp130 — base-model ESR/PSR on CogVideoX-2B

## Why
GOAL.md changes the target model: T2VUnlearning's Table 4 (the ImageNet ESR/PSR numbers we are
trying to beat) is CogVideoX-2B only — their nudity and face tables use 5B, objects do not. Every
object-thread experiment so far (exp064–exp128) ran on CogVideoX-5b, so none of it is a
same-base-model comparison against the bar in GOAL.md. Before spending any 2B training run, GOAL.md
calls for the same sanity gate exp064 ran on 5b: does the unmodified model render the ten protocol
classes, and does our restricted-convention `Original` row land near T2VUnlearning's published one
(ESR-1 21.62±20.13, ESR-5 5.09±8.23, PSR-1 78.38±2.24, PSR-5 94.91±0.92)?

## Hypothesis and what would falsify it
Hypothesis: CogVideoX-2B renders the ten Imagenette classes well enough to support the pilot (no
class near-zero top-1, the way exp064 flagged `cassette player`/`English springer`/`tench` as
taxonomy-confused rather than unrendered), and the restricted `Original` row lands in the same
ballpark as exp064's 5b row (10-way ESR-1 9.91±9.57, PSR-1 90.09±1.06) and the paper's — not
necessarily numerically close (different model, same caveat exp064 already logged about exact
agreement), but not degenerate (ESR-1 near 80 the way exp064's gate warned against, which would mean
broken prompts or a classifier/dtype problem rather than a real base-model measurement).

Falsified if: any pilot class (chain saw, church, or whichever gets attacked next) renders near-zero
top-1 on the base model, or if the restricted ESR-1/PSR-1 pair reads as degenerate rather than as a
plausible `Original` baseline. Either result blocks training on 2B until fixed — same role exp064
played for 5b.

## Setup
Exact copy of exp064's config with `model_id: THUDM/CogVideoX-2b`. Same prompts CSV
(`prompts/imagenet_objects.csv`, 200 rows, seeds baked in), same `num_frames: 49` default (GOAL.md:
frame count stays 49, not T2VUnlearning's 17 — that mismatch is already logged as ours-vs-theirs and
not something to chase), same 50 inference steps. `job_type: eval`, `mode: imagenet`, no
`lora_checkpoint_dir`, no `erased_class` — reports the whole `Original` row (mean ESR/PSR over all
ten choices of erased class) from one 200-clip generation, exactly as exp064 did.

No code change needed: `model_id` is already a plain config field threaded through
`build_eval_pipeline` (`zml/eval/eval_model.py`) with no CogVideoX-5b-specific assumption in the
eval path (checked: no hardcoded latent geometry, frame count, or VAE scaling keyed on model_id in
`zml/eval/imagenet_eval.py`). One thing NOT changed and worth flagging for the read: the pipeline
loads in `torch.bfloat16` regardless of `model_id`; the CogVideoX-2b model card's example code uses
`torch.float16` instead. bf16 should still run correctly (both clusters' GPUs support it natively),
but if this run's quality metrics look off relative to the paper, dtype is the first thing to check
before concluding 2B is a weak base model.

## What to watch
- **Per-class top-1/top-5** in `esr_psr.json` `per_class` — any class near-zero blocks it as an
  erasure target on 2B, independent of what exp064 found on 5b (exp069/exp070/exp128 already showed
  chain saw vs church split by class, not just by model).
- **`restricted` block of `esr_psr.json`**, not the top-level 1000-way one — that is the convention
  exp064 established as comparable to the paper.
- **Quality block** (clip/colorfulness/motion per class) as the reference level exp128-style checks
  will read future 2B erasure runs against.
- Whether ESR-1 lands anywhere near degenerate (tens-to-eighties the wrong way) — the sanity-gate
  failure mode exp064's own notes called out.

## Downstream
A passing gate makes this the `Original` row for every future 2B object-erasure run and unblocks a
2B `frame_replace` training job (the chain-saw-style pilot, run on 2B instead of 5b). A failing gate
is `needs_human` territory if it looks like a genuine 2B rendering weakness rather than a fixable
dtype/prompt issue.

## Results (`outputs_20260819_215924`, athena, 4h35m, job 3007742)

Restricted (10-way) `Original` row, mean over all ten leave-one-out choices of erased class:

| | ESR-1 | ESR-5 | PSR-1 | PSR-5 |
|---|---|---|---|---|
| T2VUnlearning `Original` (2B, published) | 21.62 ± 20.13 | 5.09 ± 8.23 | 78.38 ± 2.24 | 94.91 ± 0.92 |
| exp064 `Original` (5b, ours) | ~9.91 ± 9.57 | 3.44 | 90.09 ± 1.06 | 96.56 |
| **exp130 `Original` (2B, ours)** | **10.60 ± 13.30** | **2.09 ± 4.27** | **89.40 ± 1.48** | **97.91 ± 0.47** |

All four cells land on the "healthier baseline than the paper's" side, consistent with exp064's 5b
row — this is expected for an `Original` row (nothing erased) and not itself a result, but it rules
out a degenerate measurement (dtype/prompt/classifier bug), which is what this gate exists to catch.

Restricted per-class top-1 (all ten classes render well; no near-zero class):

| class | top-1 | | class | top-1 |
|---|---|---|---|---|
| tench | 0.889 | | French horn | 0.949 |
| English springer | **0.524** (weakest) | | garbage truck | 1.000 |
| cassette player | 0.905 | | gas pump | 0.853 |
| chain saw | 0.885 | | golf ball | 0.943 |
| church | **1.000** | | parachute | 0.993 |

**The ordering that mattered for the 5b thread does not obviously hold at 2B.** exp064/exp069/exp070
found chain saw the easy pilot class and church the hard scene-level one at 5b (1000-way top-1 0.513
vs 0.733, and church never erased in exp070/exp128). At 2B, restricted top-1 has church *higher* than
chain saw (1.000 vs 0.885) — church renders essentially perfectly. 1000-way top-1 (the `per_class`
block, not `restricted`) shows both classes lower on 2B than they were on 5b (chain saw 0.369 vs
0.513, church 0.549 vs 0.733), so 2B is weaker on fine-grained ImageNet taxonomy but not on rendering
the object clearly enough for the ten-way protocol. This says nothing about whether church will
erase any easier on 2B — that is a training-time question, not a rendering one — only that the base
rendering gate does not predict the 5b difficulty split.

No dtype problem surfaced: ESR-1/PSR-1 sum close to 100 as expected for a leave-one-out `Original`
row, and no per-class top-1 is near zero.

## Status
- [x] Submitted; completed 2026-08-20 (job 3007742, athena, 4h35m of a 10h allotment).
- [x] Results pulled; `Original` row recorded for 2B (table above).
- [x] Per-class weak spots noted: English springer is the only visibly weak class (0.524 restricted
      top-1); chain saw and church are both strong, unlike the 5b split.
