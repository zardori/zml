---
status: ready
concept: face
method: eval
thread: face_identity
takeaway: >
  Reported ID-Similarity (Erase/Preserve) for the Merkel frame_replace checkpoint — completes the
  pilot comparison table alongside exp090/exp091/exp097. Blocked on exp096. Not yet submitted.
---
# exp098 — reported ID-Similarity for the Merkel frame_replace LoRA

## Why
Second and final leg of the pilot's reported numbers, mirroring exp097 exactly for the second pilot
identity. Same reasoning: exp096's live eval is a training monitor at low prompt count, not the
number that goes in a table.

## Setup
Identical structure to exp097 — `mode: face`, `erased_identity: "Angela Merkel"`,
`lora_checkpoint_dir` pointing at exp096's checkpoint (fill in once chosen), same 150 published
`(prompt, seed)` pairs as exp090/exp091/exp096.

## What to watch
Same as exp097 — Erase and Preserve together, both ID-similarity conventions, `collapse_score`
against exp090's base Merkel number, and quality metrics alongside Preserve.

**The comparison that matters most here**: does Merkel's Erase/Preserve trade-off land in the same
place as Obama's (exp097)? Agreement across two demographically different identities is the strongest
evidence the split-prompt A/B/C recipe and the erase regime are genuinely identity-agnostic, not
incidentally tuned to one identity's rendering characteristics.

## Downstream
This row, plus exp090 (Original) and exp091 (NegPrompt), completes the comparison table:

| | Original (exp090) | NegPrompt (exp091) | Ours (exp097/098) |
|---|---|---|---|
| Obama Erase / Preserve | | | |
| Merkel Erase / Preserve | | | |

sitting next to T2VUnlearning's CogVideoX-5B Table 3 (Original .3853 / .3853, Erase .1158,
Preserve .2542 — their averages across all 5 identities, ours across 2).

## Status
- [ ] exp096 has a checkpoint chosen; `lora_checkpoint_dir` filled in.
- [ ] Submitted.
- [ ] Compared against exp090/exp091 and against exp097's Obama result for identity-dependence.
- [ ] Comparison table filled in and written up (`docs/face_identity.md` §6 status table).
