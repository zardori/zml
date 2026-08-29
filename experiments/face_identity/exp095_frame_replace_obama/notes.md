---
status: done
concept: face
method: frame_replace
thread: face_identity
takeaway: >
  First frame_replace erasure of a face identity (Barack Obama). `split` wins the target_variant
  grid clearly: `wholeclip` produces widespread degenerate (black/structureless) clips on both the
  erased and preserved identities mid-training, confirming the R5 motion-collapse risk. `split`
  stays clean (zero degenerate clips throughout) but reduces motion, especially on the concept
  videos. This run's small live sample could not separate identity swap from deletion; exp097's full
  evaluation and qualitative review later confirm successful erasure, usually by face deletion,
  with lower target quality and severe motion suppression but no major non-target quality loss
  beyond motion. Step 200 picked for exp096/exp097. exp096 targets Queen Elizabeth II, not Merkel
  (see `docs/face_identity.md` §6).
---
# exp095 — frame_replace erasure of Barack Obama

## Why
The question the whole pilot exists to answer: does frame_replace erase a **face identity**, and
does the temporally-spliced or whole-clip target do it better for a concept this seam-visible? Face
identity is the hardest transfer target attempted so far — present in every frame, maximally salient,
with no natural partiality (same situation as nudity, but sharper: nudity's concept region is
sometimes a body part within frame, identity is the entire face).

## Setup
Field-for-field identical to exp080's `run_002` (the human-reviewed best nudity checkpoint) except
the dataset, `concept`/`concept_target`, and the retention set — keeping the regime fixed is the
point: if identity does or doesn't erase the way nudity did, the difference is the concept, not the
hyperparameters. Regime: `erase_input_latent: original`, velocity loss, `erase_esd_eta: 2`, t in
[400, 1000), constant LR 1e-4, 200 steps, rank-8 LoRA, `gradient_accumulation_steps: 4`,
`save_interval: 20` (so step 120 — exp080's reported checkpoint step — exists here too).

**`target_variant: [split, wholeclip]`** — the one deliberate deviation from a pure field-for-field
copy, gridded into two jobs by `submit_job.py`. `split` is the temporally-spliced target (the same
mechanism nudity/objects use); `wholeclip` (`docs/face_identity.md` R4/R5) trains toward prompt A's
own plain clip → prompt B's same-seed plain clip instead, avoiding the mid-clip splice seam entirely
at the cost of being a whole-clip rewrite rather than a frame-local edit (which exp055 showed can do
broader motion damage). Both consume the merged exp115+exp116 dataset (`emit_whole_clip_target: true`
built both target types from one generation pass in each) — no extra precompute for this grid.

- Dataset: exp116's `combined_dataset/` — exp115's 9 human-kept triples merged with exp116's 43
  scale-up keeps (52 total) via `zml/precompute/merge_frame_replace_datasets.py`. Supersedes the
  original exp092 pointer (exp092 itself superseded by exp115; exp115 alone was too thin at 9/30).
- Retention: exp094's anchors (`outputs_20260811_185230`, 25 total) minus Obama's own 3
  (`retention_exclude`) — 22 anchors.

**Before submitting**, run `./merge_dataset.sh` (command in `config.yaml`'s header comment) so
`experiments/exp116_split_face_obama_dataset_scaleup/combined_dataset/` exists — the one remaining
step. exp094's real output dir is already filled in.

## What to watch
Live eval writes `summary.json` every `save_interval`; read that first, same as every other
frame_replace run — `concept_detection_rate`/`concept_area_score_mean` here are the `face_*` keys
(`docs/face_identity.md`'s naming wart: `face_detection_rate` is an *identity* rate, not a
face-presence rate — that's `face_present_rate`).

- **Erasure, both target variants.** Compare the two grid arms' final ID-sim on the concept set
  against exp090's base Obama number — both should drop meaningfully; if only one does, that decides
  which variant exp096 (Merkel) uses.
- **Shortcut test.** The concept prompts are ordinary full-identity scenes with no identity-free
  half. A drop there means the LoRA learned to remove the identity, not to copy a training clip's
  clean half.
- **`face_present_rate` on both the erased and preserved sets, every checkpoint** — per
  `docs/face_identity.md` §3.1, a collapsed rate on the *erased* set distinguishes deletion from
  identity replacement and requires qualitative review to rule out a broken clip; deletion is a
  valid erasure mechanism. A collapsed rate on the *preserved* identities is a hard fail regardless
  of what their ID-sim reads.
- **`wholeclip`-specific risk (R5):** since it's a global rewrite rather than a frame-local edit,
  watch `motion_score_mean` on preserved identities for the kind of collapse exp055 found (−84%
  concept / −29% unrelated motion, invisible to clip_score/colorfulness) — the reason a frame-local
  edit was preferred everywhere else in this project.
- **Collateral / positional-shortcut sanity** on the other four identities (`others_barack_obama.csv`,
  wired to both `control_related_prompts` and `control_unrelated_prompts` as the required-but-
  unscored-in-training slot, same convention as exp069).
- Overfitting: 52 triples per variant (exp115's 9 + exp116's 43) — still smaller than nudity's
  100-triple exp109 build, so watch for the small-dataset instability nudity's early runs (exp062
  run 2, 21 triples) hit, though 52 is past the point that run was thin at.

## Results
Both grid arms ran clean (`run_001` = split, `run_002` = wholeclip; 200/200 steps, ~6.7h each, no
errors) inside the 16h budget. Read from `summary.json`/`eval_step_*/metrics.json` in each run's
`outputs/`.

**The erasure mechanism is ambiguous by the numbers alone, on both variants.** Concept-set
`face_id_similarity_mean` and `face_detection_rate` collapse to ~0 by step 60 in both arms, but so
does concept `face_present_rate` — every checkpoint where ID-sim reads near-zero is also one where
almost no face is detected at all, indicating deletion rather than a clean "face present, wrong
identity" swap. The small live sample could not establish whether the resulting videos remained
coherent; exp097's full qualitative review later confirms that they do and treats the deletion as
successful erasure, with a clear target-quality cost.

**`wholeclip` fails on preservation.** At step 60 the *unrelated/preserved-identity* set also
collapses (`face_present_rate: 0.00`, `motion_score_mean: 0.02`, `colorfulness: 5.9`) and produces
degenerate (black/structureless) clips on both the concept set (up to 7/10 at step 80, 5/10 still at
step 200) and the preserved set (step 80). This is the R5 motion-collapse risk from exp055's
precedent, confirmed and sharper than expected — a broad quality collapse, not a targeted erasure
effect.

**`split` stays clean but reduces motion.** Zero degenerate clips at any checkpoint, either set.
Preserved-set `face_present_rate` holds at 0.42–0.66 throughout and `motion_score_mean` recovers
from an early dip (2.6→0.7 by step 100) to 1.4–1.9 by step 200. The concept set's motion score is
the caveat: it collapses early (0.9→0.03 by step 60) and only crawls back to ~0.08 by step 200 —
visible, unlike exp055's damage, in `motion_score_mean` directly rather than needing DOVER to catch
it.

**Manual review** of the pulled `eval_step_*/concept/*.mp4` clips agrees with the automated read:
`wholeclip` shows the degenerate clips clearly; `split` looks good with the same motion-reduction
caveat on concept videos, and does not cleanly resolve the identity-swap-vs-deletion ambiguity by
eye either. Checkpoint quality is hard to rank by review, but later checkpoints look best —
**step 200** is the pick for downstream use, not the step-120 default exp080 used for nudity.

**Downstream resolution (exp097):** review of the full 150-video evaluation confirms that `split`
successfully erases Obama, usually by deleting the face. Target quality visibly decreases and motion
is strongly suppressed on both target and non-target videos; apart from motion, the non-target
videos show no major quality decrease.

**Verdict: `split` wins the grid.** `wholeclip` is disqualified by the degenerate-clip rate alone,
independent of how the erasure-vs-degradation question resolves.

## Downstream
exp097 runs the full 150-video ID-Similarity eval on `run_001` (split) step 200 — the live numbers
above are a progress signal, not the reported metric (same relationship exp069→exp071 has).
exp096 (Queen Elizabeth II) uses `target_variant: split`, not a repeated grid.

## Status
- [x] exp115/exp116 (dataset) and exp094 (retention) complete; timestamps filled in. Only
      `./merge_dataset.sh` (builds `combined_dataset/` on helios) is left before submitting.
- [x] Submitted (2-job grid: split, wholeclip) — `grid_20260814_141010`.
- [x] Both variants compared on erasure + preservation + `face_present_rate`; manual review agrees.
      `split` wins for exp096; `wholeclip` disqualified by widespread degenerate clips.
