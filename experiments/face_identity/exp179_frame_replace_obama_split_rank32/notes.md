---
status: done
concept: face
method: frame_replace
thread: face_identity
takeaway: >
  Rank 32 improves exp095's live erasure/quality trade-off at step 180: Obama detection remains
  0/10, but faces survive in 6/10 target clips instead of 1/10 and target motion rises 0.080 ->
  0.129, with no degenerate clips. Step 200 regresses to 3/10 face-containing clips, motion 0.042,
  and 1/10 degenerate, so step 180 is selected for exp181's full 150-prompt evaluation.
submitted: 2026-08-29 15:59 helios job 21416924
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

## Results
The successful rerun completed all 200 steps on helios in 6h49m
(`outputs_20260829_155907`; the earlier `outputs_20260829_144553` attempt failed after 31 seconds
because one merged-dataset latent was missing). Training health is clean: no health flags, with
recent total loss 0.4792 versus 0.5993 initially.

**Step 180 is the winner, not the final checkpoint.** Against exp095 rank-8 split step 200 on the
same live protocol, it keeps thresholded Obama detection at 0.00 while improving target
`face_present_rate` 0.10 -> 0.50 (faces in 1/10 -> 6/10 clips), target motion 0.080 -> 0.129, and
colorfulness 30.64 -> 61.27. Target CLIP score is unchanged (0.29 -> 0.30), and both checkpoints
have zero degenerate clips. The six face-containing clips with zero Obama identifications are the
first live evidence that higher rank may shift the mechanism from face deletion toward identity
replacement.

Preservation also improves numerically at step 180: unrelated face-present rate 0.66 -> 0.69,
motion 1.87 -> 2.53, and CLIP score 0.33 -> 0.34, with no degenerate clips. That set contains only
four videos, however, so it is a direction rather than a reported preservation result.

Step 200 regresses on the target side despite retaining 0.00 Obama detection: face-present rate
falls to 0.33, motion to 0.042, and one of ten target clips is degenerate. It is not selected.

The result remains an n=10 checkpoint monitor. exp181 repeats exp097's full 150-prompt protocol at
step 180 to determine whether the apparent identity-replacement and motion improvements survive.

## Status
- [x] Submit (project owner) — `./submit_job.py helios experiments/face_identity/exp179_frame_replace_obama_split_rank32/config.yaml`
- [x] Compared against exp095's rank-8 split arm on erasure, preservation, and concept-motion
      collapse; step 180 selected.
- [ ] Full 150-prompt evaluation and qualitative review (exp181).
