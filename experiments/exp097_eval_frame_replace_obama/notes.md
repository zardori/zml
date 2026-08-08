---
status: ready
concept: face
method: eval
thread: face_identity
takeaway: >
  Reported ID-Similarity (Erase/Preserve) for the Obama frame_replace checkpoint — the number that
  goes in the final comparison table, not exp095's live-eval monitor. Blocked on exp095. Not yet
  submitted.
---
# exp097 — reported ID-Similarity for the Obama frame_replace LoRA

## Why
exp095's live eval runs at `eval_num_prompts: 10` per checkpoint as a training monitor, not a
publishable result — the same lesson exp082 established for nudity (n=10 is too weak to distinguish
anything; exp073's whole five-checkpoint trajectory was consistent with pure noise at that n). The
published metric needs all 150 prompts (30 for Obama, 120 for the other four, matched exactly to
exp090's base-model run and exp091's NegPrompt run) generated fresh by the *unlearned* model.

## Setup
`mode: face`, `erased_identity: "Barack Obama"`, `lora_checkpoint_dir` pointing at exp095's winning
`target_variant` checkpoint — **fill in the specific step** once exp095 completes and a checkpoint is
chosen (exp080's precedent is step 120 out of 200; not assumed here without exp095's own numbers).

Same `(prompt, seed)` pairs as exp090 and exp091 — identical to how nudity's exp082/exp083 measure on
the same I2P/SafeSora pairs so only the intervention differs across the three rows.

## What to watch
Same reading as every other reported eval in this project:
- **Erase and Preserve together**, not Erase alone — a low Erase with a collapsed
  `face_present_rate` is degradation, not erasure (`docs/face_identity.md` §3.1's hard rule).
- **`collapse_score`** on the erased identity's own 30 clips (recorded in `id_similarity.json`) —
  compare against exp090's base-model collapse_score for Obama; a large jump is R6's
  fixed-substitute-collapse failure mode (the LoRA learned one specific replacement face, not
  removal in general).
- **Both conventions** (face-conditioned headline vs. `zerofill`) — report both, per
  `docs/face_identity.md` §3.1.
- Quality (`clip_score`/`colorfulness`/`motion_score`) alongside Preserve, watching specifically for
  the `wholeclip`-variant motion-collapse risk (R5) if that's the variant that won exp095's grid.

## Downstream
This row, plus exp090 (Original) and exp091 (NegPrompt), fills the Obama column of the comparison
table sitting next to T2VUnlearning's CogVideoX-5B Table 3 block.

## Status
- [ ] exp095 has a checkpoint chosen; `lora_checkpoint_dir` filled in.
- [ ] Submitted.
- [ ] Compared against exp090 (Original) and exp091 (NegPrompt) — Erase, Preserve,
      `face_present_rate`, `collapse_score`, and quality, not Erase alone.
