---
status: active
concept: face
method: frame_replace
thread: face_identity
takeaway: >
  Not yet run. Rerun of exp095's `split` arm (the winner of that grid) at lora_rank 32 instead of
  rank 8, to see whether more LoRA capacity changes the identity-swap-vs-face-deletion ambiguity or
  the motion reduction on concept videos that exp095 flagged.
submitted: 2026-08-29 14:45 helios job 21415841
---
# exp179 — frame_replace erasure of Barack Obama, split target, rank 32

## Why
exp095 gridded `target_variant: [split, wholeclip]` at `lora_rank: 8` and found `split` wins cleanly
(`wholeclip` produces widespread degenerate clips). This run isolates the next knob — LoRA rank —
on top of that winning variant, matching the rank-32 reruns already done for nudity (exp129, exp136)
and imagenet objects (exp141, exp142, exp160): does more capacity sharpen erasure, resolve the
face-present-but-wrong-identity ambiguity exp095 couldn't settle, or recover the concept-set motion
collapse exp095 saw by step 200 (0.9 → ~0.08)?

## Setup
Field-for-field identical to exp095 except:
- `target_variant: split` only (not gridded — `wholeclip` is already disqualified, no need to rerun
  it).
- `lora_rank: 32`, `lora_alpha: 32.0` (alpha = rank, per the project's rank-32 convention in
  exp129/exp136/exp141/exp142/exp160), vs. exp095's rank 8 / alpha 8.

Same dataset (exp116's `combined_dataset/`, 52 triples), same retention set (exp094 minus Obama's
own anchors), same erase regime (`erase_esd_eta: 2`, velocity loss, t in [400, 1000), constant LR
1e-4, 200 steps, `gradient_accumulation_steps: 4`, `save_interval: 20`).

## What to watch
Same checklist as exp095 (`docs/face_identity.md` §3.1): read `summary.json` first.
- Concept-set `face_id_similarity_mean` / `face_detection_rate` vs. `face_present_rate` — exp095's
  ambiguity was that both collapsed together at step 60, so ID-sim ~0 never confirmed an identity
  swap rather than face deletion. Check whether rank 32 changes this.
- Preserved-identity `face_present_rate` and `motion_score_mean` — exp095's rank-8 split arm held
  clean (no degenerate clips) but dipped early before recovering by step 200.
- Concept-set `motion_score_mean` — exp095's unresolved caveat (collapsed to ~0.03 by step 60, only
  crawled to ~0.08 by step 200). Watch whether extra capacity makes this worse (more aggressive
  overfit to a frozen-frame shortcut) or better.
- Degenerate/black clip rate on both sets, every checkpoint — the `wholeclip` failure mode; confirm
  `split` still avoids it at higher rank.

## Status
- [x] Submit (project owner) — `./submit_job.py helios experiments/face_identity/exp179_frame_replace_obama_split_rank32/config.yaml`
- [ ] Compare against exp095's rank-8 split arm on erasure, preservation, and concept-motion collapse.
