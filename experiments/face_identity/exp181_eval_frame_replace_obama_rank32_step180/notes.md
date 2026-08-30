---
status: active
concept: face
method: eval
thread: face_identity
takeaway: >
  Full 150-prompt evaluation of exp179's rank-32 Obama frame_replace checkpoint at step 180. The
  live n=10 monitor suggests the same complete erasure as exp095 rank 8, but with substantially
  more target faces and motion; this run tests that apparent identity-replacement improvement on
  the reported protocol. Not yet submitted.
submitted: 2026-08-30 17:59 helios job 21449712
---
# exp181 — full evaluation of rank-32 Obama frame_replace, step 180

## Why
exp179 reruns exp095's winning `split` method at LoRA rank 32. Its live evaluation is encouraging,
but contains only 10 Obama prompts and 4 preservation prompts, so it cannot support a reported
comparison. This experiment mirrors exp097's full evaluation: all 150 published face prompts, with
30 prompts for Obama and 30 for each of the four preserved identities, using the same prompt/seed
pairs as the base-model, NegPrompt, and rank-8 evaluations.

The main question is whether rank 32 changes the erasure mechanism. Exp097 found that rank 8 erased
Obama mostly by deleting the face: only 2/30 target clips retained any detectable face. At exp179
step 180, the live monitor finds faces in 6/10 target clips while Obama identification remains 0/10.
If that survives the full evaluation and qualitative review, rank 32 is a materially cleaner
identity replacement rather than merely another deletion-based eraser.

## Setup
The config is field-for-field identical to exp097 except for the checkpoint. It evaluates
`exp179`'s `frame_replace_lora_step180` with `mode: face`, `erased_identity: "Barack Obama"`, 50
inference steps, and `prompts/face_cogvideox.csv`.

Step 180 is selected over exp179's final step 200 from the live metrics:

| target metric | step 180 | step 200 |
|---|---:|---:|
| Obama detection rate | 0.00 | 0.00 |
| face-present rate | 0.50 | 0.33 |
| clips without a face | 4/10 | 7/10 |
| motion score | 0.129 | 0.042 |
| degenerate clips | 0/10 | 1/10 |

Source: `experiments/face_identity/exp179_frame_replace_obama_split_rank32/outputs_20260829_155907/`
`eval_step_{180,200}/metrics.json`. Step 180 also improves
the n=10 live monitor over exp095 rank-8 step 200: target face-present rate 0.50 vs 0.10 and target
motion 0.129 vs 0.080, with 0% Obama detection and no degenerate clips in both.

## What to watch
- **Erase and Preserve:** compare face-conditioned and zerofill ID similarity directly with exp090
  (base), exp091 (NegPrompt), and exp097 (rank-8 frame_replace).
- **Erasure mechanism:** target `face_present_rate`, `clips_without_face`, and `identified_rate`.
  The rank-32 hypothesis requires low Obama identification while retaining substantially more faces
  than exp097's 2/30 clips.
- **Replacement diversity:** inspect `collapse_score` and the surviving faces for a repeated fixed
  substitute. A higher face-present rate is not an improvement if every prompt receives the same
  replacement identity.
- **Target quality and motion:** compare CLIP score, colorfulness, motion, and degenerate clips with
  exp090 and exp097. The live result suggests improvement but target motion is still very low.
- **Preservation:** check every non-target identity separately. The live preservation sample has
  only four videos and cannot establish that rank 32 avoids collateral damage.
- **Qualitative review:** review the Obama clips with detectable faces and a sample from each
  preserved identity. The numerical distinction between face replacement, deletion, and malformed
  faces needs visual confirmation.

DOVER will read `0.0` on helios and must be treated as unmeasured, per `CLAUDE.md`; score the pulled
videos post-hoc on x86_64 if a technical-quality comparison is needed.

## Downstream
Compare the resulting row directly with exp097. If the full set confirms zero or near-zero Obama
identification with materially higher face presence and motion, exp179 step 180 supersedes exp095
step 200 as the Obama checkpoint. Otherwise, retain exp097 as the reported result and treat the
live improvement as small-sample noise.

## Status
- [x] exp179 completed and step 180 selected from the live trajectory.
- [x] Configured the same 150-prompt protocol as exp097.
- [ ] Submitted by a project owner.
- [ ] Results compared with exp090, exp091, and exp097.
- [ ] Human review completed, especially the target clips retaining faces.
- [ ] `docs/face_identity.md` updated if rank 32 changes the reported conclusion.
