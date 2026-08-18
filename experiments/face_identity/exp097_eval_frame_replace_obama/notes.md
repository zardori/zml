---
status: done
concept: face
method: eval
thread: face_identity
takeaway: >
  THE OBAMA ROW: Erase 0.0497 against base 0.5081 and NegPrompt 0.3391, with Preserve 0.4205
  actually ABOVE base's 0.3846 — by the headline numbers the strongest erasure in this project, and
  identity collateral is nil. But §3.1's hard rule fires: Obama's `face_present_rate` collapses
  0.8735 -> 0.0714 (28 of 30 clips contain no detectable face), so this is face *deletion*, not
  identity removal, and the Erase figure is not evidence of the latter. That settles exp095's open
  question, which manual review could not. Motion also collapses globally — the four preserved
  identities lose a mean 76% (Obama -93%) — matching exp071 (objects, -45%) and exp111 (nudity,
  -89%) the same week. Report Erase only alongside face_present_rate.
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

## Results (2026-08-15) — the number lands, and §3.1's hard rule fires

Completed on helios in 5.3 h, all 150 prompts, `exp095 run_001 frame_replace_lora_step200`,
`identity_threshold` 0.23. Same `(prompt, seed)` pairs as exp090 and exp091.

### The Obama column

| | Erase (id_sim) ↓ | Preserve ↑ | Obama face_present | Obama identified_rate |
|---|---|---|---|---|
| base (exp090) | 0.5081 | 0.3846 | 0.8735 | 0.8667 |
| NegPrompt (exp091) | 0.3391 | 0.3460 | 0.8434 | 0.7333 |
| **frame_replace (ours)** | **0.0497** | **0.4205** | **0.0714** | **0.0000** |

Under the `zerofill` convention: ours Erase **0.0035** / Preserve 0.3778, NegPrompt 0.2860 / 0.2824.
Both conventions agree, as §3.1 requires them to be reported.

**Preserve is above base** (0.4205 vs 0.3846), and every preserved identity's `face_present_rate`
*rises* against base (Merkel 0.674 -> 0.838, Trump 0.805 -> 0.958, Biden 0.788 -> 0.829, Elizabeth
0.820 -> 0.949). There is no identity collateral at all — the LoRA does not touch the other four.

### Why the Erase number cannot be reported alone

`docs/face_identity.md` §3.1's hard rule: *a low Erase with a collapsed `face_present_rate` is
degradation, not erasure.* Obama's collapses from 0.8735 to **0.0714** — **28 of 30 clips contain no
detectable face at all**. The model has not learned to render a different person for "Barack Obama";
it has learned to render no face.

`collapse_score` 0.0711 against base 0.5782 looks like a pass on R6 (fixed-substitute collapse), but
it is computed over face embeddings and there are almost no faces to embed, so it carries no
information here. R6 is untested by this run, not cleared by it.

**This settles exp095's open question.** exp095's manual review agreed something collapsed but "can't
cleanly separate identity-swap from face-deletion". At 28/30 clips without a face, on 30 prompts
rather than exp095's 10, it is face-deletion.

### Motion collapses globally, as in the other two threads

| identity | base motion | ours | delta |
|---|---|---|---|
| Barack Obama *(erased)* | 1.362 | 0.097 | **-93%** |
| Angela Merkel | 0.905 | 0.158 | -83% |
| Donald Trump | 0.710 | 0.157 | -78% |
| Joe Biden | 0.777 | 0.194 | -75% |
| Queen Elizabeth II | 0.716 | 0.229 | -68% |

**Mean over the four preserved identities: -76%.** So identity semantics are preserved while motion
is not — the same dissociation exp071 found for objects (PSR intact, -45% motion on preserved
classes) and exp111 for nudity (-89% on the safe related set). Three concepts, one failure mode.

NegPrompt, by contrast, runs Obama motion at 1.734, *above* base. It erases far less and damages
nothing.

Clip score on Obama drops 0.3502 -> 0.3012 and colorfulness 49.8 -> 46.1; on the preserved identities
clip score is 0.330-0.374 against base — essentially unharmed.

## Status
- [x] exp095 has a checkpoint chosen (`split`, step 200); `lora_checkpoint_dir` filled in.
- [x] Submitted and complete (helios, 5.3 h, `outputs_20260815_125349`).
- [x] Compared against exp090 (Original) and exp091 (NegPrompt) on Erase, Preserve,
      `face_present_rate`, `collapse_score` and quality.
- [ ] **Human review of the 28 face-less Obama clips** — what is rendered instead (back of head?
      occlusion? empty scene?) decides whether this is reportable as erasure at all.
- [ ] An earlier checkpoint than step 200 measured: step 200 was picked on exp095's n=10 monitor, and
      exp112/exp071 both showed that set cannot rank checkpoints. A step where `face_present_rate`
      still holds is the checkpoint worth reporting.
- [ ] `docs/face_identity.md` updated with the row and the face-deletion caveat.
