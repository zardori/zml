---
status: ready
concept: face
method: eval
thread: face_identity
takeaway: >
  Base-model ID-Similarity reference on all 5 identities — the `Original` row, the pilot's hard
  gate, and the source of `identity_threshold`. Not yet submitted.
---
# exp090 — base-model ID-Similarity on the face-identity protocol

## Why
Third comparison axis after nudity and ImageNet objects (`docs/comparison_targets.md` §2.3):
T2VUnlearning §4.3 evaluates face erasure on **CogVideoX-5B**, the same base model we use — unlike
the ImageNet axis, this is a true like-for-like row, no "different base model" caveat needed. Their
150 eval prompts are published (`prompts/face_cogvideox.csv`, fetched verbatim via
`tools/fetch_face_eval_prompts.py`), so we measure on a set nobody on the team wrote — see
`docs/external_eval_sets.md` §1 for why that matters. Full protocol, the crux around no-face frames,
and the deviations table: `docs/face_identity.md`.

This run is the pilot's hard gate. Nothing downstream (exp091–exp098) is worth building unless it
passes.

## Setup
`mode: face`, no `lora_checkpoint_dir`, no `erased_identity` → the unmodified base model, scored
against all 5 identities. `zml.eval.face_eval.main` generates 30 clips per identity into
`eval_step_0/<slug>/video_{i}.mp4` (resumable — an existing non-empty file is not regenerated) and
scores every clip's ArcFace cosine similarity to *its own* identity's reference embedding
(`zml/benchmarks/data/face_reference_embeddings.json`, built by
`tools/build_face_reference_embeddings.py` from freely-licensed Wikimedia photos — provenance and
the two build-time gates in `docs/face_identity.md` §4.2).

With `erased_identity` unset, `_leave_one_out_report` fills the whole `Original` row from this one
150-video run: Erase/Preserve computed with each identity in turn as the hypothetical erased one,
plus mean/std across the five (mirrors `imagenet_eval`'s `Original` row).

Their CogVideoX-5B row, for comparison — not a target to match (see What to watch):

| | Merkel | Obama | Trump | Biden | Elizabeth | AVG |
|---|---|---|---|---|---|---|
| Original | .3379 | .4362 | .3547 | .3267 | .4710 | **.3853** |

## What to watch
- **The gate criteria, not agreement with .3853.** Their embedder, detector and reference set are
  all unpublished, so our absolute scale need not match theirs even on the base model. Pass/fail is:
  (a) at least two identities with face-conditioned ID-sim ≥ ~0.30, (b) `face_present_rate` ≥ 0.8 on
  those identities, (c) clean separation between right-identity and wrong-identity distributions —
  score every identity's 30 clips against all 5 references (not just their own) and look at the
  resulting 5x5 matrix. If our Original lands far from theirs, say so in the write-up; do not tune
  the instrument to match (the same policy `docs/imagenet_objects.md` §3.1 follows for the
  1000-way/10-way ambiguity).
- **`identity_threshold` calibration.** The 5x5 cross-reference matrix from the point above is what
  sets it — calibrated against the negative distribution exactly as `docs/imagenet_objects.md` §5
  calibrates `frame_concept_threshold`. The value shipped in `check_for_face.IDENTITY_THRESHOLD`
  (0.30) is an explicit placeholder; do not treat `face_detection_rate` from any run as meaningful
  until this is set for real.
- **`face_present_rate` per identity.** Low ID-sim with a collapsed face rate is degradation of the
  *base model's own rendering*, not evidence the metric is broken — but it changes which identities
  are viable pilot targets (see next point).
- **Which two identities for the pilot** (exp092/exp093). Rule: the highest base-model ID-sim
  identity, plus the highest-scoring identity that differs from it demographically. Expected from
  the paper's Original column: **Obama** (.4362) + **Merkel** (.3379, non-US and female) — but this
  run's actual numbers decide, not the paper's.
- **`collapse_score` at the base model.** Recorded per identity in `id_similarity.json` even though
  collapse is mainly an erased-model failure mode (R6) — useful as the reference point exp095/096's
  own collapse numbers get compared against.
- Per `docs/face_identity.md` §3.1, visually spot-check a handful of clips before trusting the
  numbers — same discipline every other detector in this project gets.

## Downstream
Sets `identity_threshold` for exp091–exp098, picks the 2 pilot identities for exp092/093/095/096,
and is the `Original` row of the final comparison table.

## Status
- [ ] `tools/fetch_face_models.py` run on the target cluster's login node.
- [ ] Submitted.
- [ ] Gate criteria checked (base ID-sim, face_present_rate, 5x5 separation matrix).
- [ ] `identity_threshold` calibrated and recorded here.
- [ ] Pilot identities confirmed (expected: Obama + Merkel, pending this run's numbers).
