---
status: done
concept: face
method: eval
thread: face_identity
takeaway: >
  THE BAR, and it is a low one. NegPrompt moves Erase 0.5081 -> 0.3391 for Obama and 0.3272 ->
  0.2234 for Elizabeth — both still above the 0.23 identification threshold, and identified_rate
  only falls 0.867 -> 0.733 and 0.600 -> 0.400. It also costs Preserve (0.3846 -> 0.3460 for the
  Obama run), i.e. it pulls the four identities it was not aimed at further from base than the one it
  was. What it does NOT do is damage the video: face_present_rate holds (0.843 / 0.800) and motion is
  at or above base, so its weak erasure is honest weakness rather than collapse — the opposite trade
  from exp097's. Same pattern as exp065 for objects: the training-free baseline distorts without removing.
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

## Results (2026-08-12) — a weak but honest baseline

Two runs, ~4.7 h each on helios, all 150 prompts at `identity_threshold` 0.23, same `(prompt, seed)`
pairs as exp090.

| erased identity | Erase ↓ | Preserve ↑ | erased face_present | erased identified_rate |
|---|---|---|---|---|
| Barack Obama — base (exp090) | 0.5081 | 0.3846 | 0.8735 | 0.8667 |
| **Barack Obama — NegPrompt** | **0.3391** | 0.3460 | 0.8434 | 0.7333 |
| Queen Elizabeth II — base (exp090) | 0.3272 | — | 0.8197 | 0.6000 |
| **Queen Elizabeth II — NegPrompt** | **0.2234** | — | 0.8000 | 0.4000 |

`zerofill` convention, Obama run: Erase 0.2860 / Preserve 0.2824.

### It suppresses, it does not remove

Obama's Erase falls by a third (0.5081 -> 0.3391) but stays **above** the 0.23 identification
threshold, and 22 of 30 clips are still identified as Obama. Elizabeth's 0.2234 does cross below the
threshold, but she starts at 0.3272 — barely above it — so that crossing is cheap and her
identified_rate still sits at 0.400.

This is the face-thread twin of exp065: the training-free baseline visibly distorts the target
without removing it, which is what T2VUnlearning attributes to NegPrompt and what our method has to
beat to be worth its training cost.

### It is not degrading the model, which matters for the comparison

`face_present_rate` on the erased identity holds at 0.8434 (Obama) and 0.8000 (Elizabeth) against
base's 0.8735 / 0.8197, and Obama-prompt motion runs **1.734 against base 1.362** — above base.
So §3.1's degradation caveat does *not* apply to this row: NegPrompt's Erase number means what it
says.

That is the precise contrast with [exp097](../exp097_eval_frame_replace_obama/notes.md), which
reaches Erase 0.0497 but with `face_present_rate` at 0.0714 and motion at -93%. The two rows sit at
opposite ends of the same trade: NegPrompt keeps the video and fails to erase; frame_replace erases
and loses the video.

### The collateral is real but small

Preserve drops 0.3846 -> 0.3460 on the Obama run — the four identities NegPrompt was *not* aimed at
lose more similarity than they should, presumably because "Barack Obama" as a negative prompt pushes
against generic political-portrait content. Ours holds Preserve at 0.4205, above base, so on this
axis frame_replace is cleanly better.

## Status
- [x] exp090 complete; pilot identities confirmed as Obama + Queen Elizabeth II (`erased_identity`
      updated from the Obama/Merkel placeholder).
- [x] Submitted and complete (grid, 2 jobs, 2026-08-11/12, ~4.7 h each on helios).
- [x] Compared against exp090 on Erase, Preserve and quality.
- [ ] `docs/face_identity.md` comparison table updated with both rows.
