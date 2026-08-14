---
status: ready
concept: face
method: frame_replace
thread: face_identity
takeaway: >
  First frame_replace erasure of a face identity (Barack Obama), copying exp080's best nudity
  regime field-for-field. Grid target_variant [split, wholeclip] — the A/B this run exists to
  answer. exp116 (dataset scale-up) and exp094 (retention) are both done; only the
  merge_dataset.sh step (builds exp116's combined_dataset/ on helios) is left before this can be
  submitted.
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
`experiments/face_identity/exp116_split_face_obama_dataset_scaleup/combined_dataset/` exists — the one remaining
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
  `docs/face_identity.md` §3.1, a collapsed face rate on the *erased* set alongside a low ID-sim
  means degradation, not erasure; a collapsed rate on the *preserved* identities is a hard fail
  regardless of what their ID-sim reads.
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

## Downstream
exp097 runs the full 150-video ID-Similarity eval on whichever checkpoint(s) look best here — the
live numbers are a progress signal, not the reported metric (same relationship exp069→exp071 has).
exp096 (Merkel) uses whichever `target_variant` wins here, not a repeated grid.

## Status
- [x] exp115/exp116 (dataset) and exp094 (retention) complete; timestamps filled in. Only
      `./merge_dataset.sh` (builds `combined_dataset/` on helios) is left before submitting.
- [ ] Submitted (2-job grid: split, wholeclip).
- [ ] Both variants compared on erasure + preservation + `face_present_rate`; a winner picked for
      exp096.
