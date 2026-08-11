---
status: ready
concept: face
method: eval
thread: face_identity
takeaway: >
  NegPrompt baseline for the 2 pilot identities (Obama + Queen Elizabeth II, confirmed by exp090) —
  the training-free bar our method has to beat. Not yet submitted.
---
# exp091 — NegPrompt baseline (face identity)

## Why
A reviewer's first objection to any erasure method is that a training-free negative prompt might do
the job for free. Nudity's exp083 found NegPrompt is a genuinely strong baseline there (-68%/-52% on
I2P/SafeSora at no measurable quality cost) — the same test needs to exist for faces before any
frame_replace result is claimed as meaningful, and T2VUnlearning provides no baseline row to compare
against for this axis at all.

## Setup
Identical to exp090 (same base model, same 150 published prompts, same per-prompt seeds) with
exactly one difference: `negative_prompt: auto`, which resolves to each grid arm's `erased_identity`
— the model is told not to render that person's name, for every one of that identity's own 30
prompts. No training, no LoRA.

`erased_identity: ["Barack Obama", "Queen Elizabeth II"]` grids into two jobs, one per pilot identity
— exp090's actual gate numbers (not the pre-run Obama + Merkel guess) picked this pair: Obama is the
highest-id_sim identity, Merkel the lowest (and the most degenerate-clip-affected — see exp090's
notes.md), so Elizabeth (next-highest, and demographically distinct from Obama) is the corrected
second pilot identity.

## What to watch
- **Erase vs Preserve, not Erase alone.** `zml.eval.face_eval.score_existing` still scores all 5
  identities even though the negative prompt only targets one — so Preserve (the other 4 identities,
  *not* negative-prompted) should sit close to exp090's base numbers. A collateral drop on identities
  the negative prompt never touched would be a NegPrompt-specific finding worth flagging, the same
  way exp083 found unrelated-prompt colorfulness moved +16% under nudity's negative prompt despite
  CLIP score reporting no change.
- **`face_present_rate` on the erased identity's own clips.** Per `docs/face_identity.md` §3.1, a low
  Erase score with a collapsed face rate is degradation (the negative prompt suppressing faces
  generally), not erasure (suppressing *this* identity specifically) — visually distinguishable, not
  distinguishable from the ID-sim number alone.
- Quality (`clip_score`/`colorfulness`/`motion_score`) on both the erased identity's set and the
  other four's, mirroring exp083's collateral table.

## Downstream
The number our frame_replace checkpoints (exp095–exp098) must beat.

## Status
- [x] exp090 complete; pilot identities confirmed as Obama + Queen Elizabeth II (`erased_identity`
      updated from the Obama/Merkel placeholder).
- [ ] Submitted.
- [ ] Compared against exp090 on Erase, Preserve, and quality — not Erase alone.
