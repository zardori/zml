---
status: ready
concept: nudity
method: eval
thread: nudity
takeaway: >
  The whole comparable table re-measured on exp110's step-140 checkpoint, which beat the old
  incumbent on every axis at equal erasure. Covers exp102 + exp084's four concept sets in one
  submission, plus the paired safe `related` set whose baselines exp111 just established. The most
  interesting cell is not an erasure rate: it is whether exp110 fixes the -89% motion loss exp111
  found on nudity-free safe rewrites. 4 jobs. Human review of the checkpoint still pending.
---
# exp112 — comparable table on the gen4 checkpoint

## Why
[exp110](../exp110_frame_replace_nudity_gen4/notes.md) step 140 reads rate 0.0000 at colorfulness
35.4 (base 36.3) and motion 0.25, against the old incumbent exp080 r2 s120's 0.0000 / 21.9 / 0.11.
Every number in [`docs/comparability_t2vunlearning.md`](../../docs/comparability_t2vunlearning.md)
is therefore measured on a superseded checkpoint.

This runs exp102's and exp084's coverage together so the table moves in one piece: Gen (100),
Ring-A-Bell (79), I2P (95), SafeSora (100), plus `related` and `unrelated`.

## The cell that matters most
Not the erasure rates — those are expected to hold, since the live eval already reads 0.0000 on 490
frames of the Gen set. It is **`related`**. [exp111](../exp111_related_baselines_safe_set/notes.md)
established base 0.130 / NegPrompt 0.050 there, and found the old checkpoint froze that set as
badly as the concept set itself (motion 0.37 -> 0.04, **-89%**, on prompts containing no nudity).
That gradient — -93% concept, -89% related, -68%/-36% VBench, -19% fire-unrelated — is the honest
form of the preservation story.

exp110 holds 2.3x the motion on concept prompts. If that carries to `related`, the gradient flattens
and "graded, not confined" softens into something much easier to defend. If it does not, the new
checkpoint is better on the concept set and no better where it matters for preservation.

## Checkpoint choice
**step 140**, which dominates step 120 on metrics: identical rate (0.0000) and motion (0.25), 4
points more colorfulness, same clip score. The one thing that would reverse it is if 140's extra
colour is artefacts rather than saturation — DOVER answers that and was still scoring locally when
this was staged. If it flags 140, change `lora_checkpoint_dir` to `...step120` before submitting.

## Standing caveat
**Human review has not happened.** The clips are pulled and staged for it. Running the eval now is
fine — it costs hours and unblocks the table — but per [[feedback-detector-metrics-not-ground-truth]]
nothing here is reportable until the checkpoint has been watched. If review rejects step 140 this is
a one-field re-run.

## Status
- [ ] Submitted (4 jobs).
- [ ] Human review of exp110 step 140 (independent of this run, but gates reporting).
- [ ] DOVER scored locally on the outputs (helios writes 0.0).
- [ ] `docs/comparability_t2vunlearning.md` §4 rewritten on the new checkpoint.
