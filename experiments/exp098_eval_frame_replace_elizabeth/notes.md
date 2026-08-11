---
status: ready
concept: face
method: eval
thread: face_identity
takeaway: >
  Reported ID-Similarity (Erase/Preserve) for the Queen Elizabeth II frame_replace checkpoint —
  completes the pilot comparison table alongside exp090/exp091/exp097. Second pilot identity
  confirmed by exp090 (supersedes this experiment's original Angela Merkel target — see exp090's
  notes.md). Blocked on exp096. Not yet submitted.
---
# exp098 — reported ID-Similarity for the Queen Elizabeth II frame_replace LoRA

## Why
Second and final leg of the pilot's reported numbers, mirroring exp097 exactly for the second pilot
identity. Same reasoning: exp096's live eval is a training monitor at low prompt count, not the
number that goes in a table.

This experiment originally targeted Angela Merkel on the pre-run guess applied to T2VUnlearning's
published numbers; exp090's actual base-model numbers found Merkel the weakest identity of all five
(see exp090's notes.md), so Queen Elizabeth II is the corrected pilot identity.

## Setup
Identical structure to exp097 — `mode: face`, `erased_identity: "Queen Elizabeth II"`,
`lora_checkpoint_dir` pointing at exp096's checkpoint (fill in once chosen), same 150 published
`(prompt, seed)` pairs as exp090/exp091/exp096.

## What to watch
Same as exp097 — Erase and Preserve together, both ID-similarity conventions, `collapse_score`
against exp090's base Elizabeth number (.3272 face-conditioned, `face_present_rate` .8474 post
degenerate-frame fix), and quality metrics alongside Preserve.

**The comparison that matters most here**: does Elizabeth's Erase/Preserve trade-off land in the same
place as Obama's (exp097)? Agreement across two demographically different identities is the strongest
evidence the split-prompt A/B/C recipe and the erase regime are genuinely identity-agnostic, not
incidentally tuned to one identity's rendering characteristics.

## Downstream
This row, plus exp090 (Original) and exp091 (NegPrompt), completes the comparison table:

| | Original (exp090) | NegPrompt (exp091) | Ours (exp097/098) |
|---|---|---|---|
| Obama Erase / Preserve | .5082 / — | | |
| Elizabeth Erase / Preserve | .3272 / — | | |

sitting next to T2VUnlearning's CogVideoX-5B Table 3 (Original .3853 avg, Erase .1158, Preserve
.2542 — their averages across all 5 identities, ours across 2). Note their per-identity Original
ranking disagrees with ours on Elizabeth specifically — their highest (.4710) is our second-lowest
(.3272) — so this row's absolute scale is not expected to land near theirs; the gate criteria
(`docs/face_identity.md` §2), not agreement with their number, is what to check.

## Status
- [x] exp090 confirms Queen Elizabeth II as the second pilot identity (not Merkel — see Why).
- [ ] exp096 has a checkpoint chosen; `lora_checkpoint_dir` filled in.
- [ ] Submitted.
- [ ] Compared against exp090/exp091 and against exp097's Obama result for identity-dependence.
- [ ] Comparison table filled in and written up (`docs/face_identity.md` §6 status table).
