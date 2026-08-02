---
status: active
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Reported ESR/PSR row for the church LoRA. With exp071 it decides whether the object pilot
  succeeded and whether the remaining eight classes are worth running.
---
# exp072 — reported ESR/PSR for the church LoRA

## Goal
The `frame_replace (ours)` row for church, on the same 200 prompts and seeds as exp064/exp065.
Together with exp071 this decides whether the pilot succeeded and whether it is worth running the
remaining eight classes.

## Setup
Identical to exp071 with `erased_class: "church"` and the exp070 checkpoint. See exp071's note on
choosing a checkpoint deliberately rather than picking the best-scoring one.

`./submit_job.py athena experiments/exp072_eval_frame_replace_church/config.yaml`

## What to watch
Same reads as exp071, plus the pilot's actual question: **how far apart are chain saw and church?**
A large gap says frame_replace depends on the concept being localized, which is a real finding about
the method and belongs in `docs/frame_replace.md`, not just in this protocol's write-up.

## Decision this feeds
- Both classes erase with acceptable PSR -> run the remaining eight classes and report the full
  table against T2VUnlearning.
- Chain saw works, church does not -> report the pilot honestly, and either restrict the claim to
  localized concepts or investigate the scene-level case (split_step_frac, larger datasets).
- Neither works -> the split-prompt -> frame_replace chain does not transfer to objects; revisit
  before spending on the other eight.

## Status
- [ ] exp070 complete; checkpoint chosen and recorded here.
- [ ] Submitted.
- [ ] Row added; pilot decision recorded in `docs/imagenet_objects.md`.
