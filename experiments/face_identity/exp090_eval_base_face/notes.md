---
status: done
concept: face
method: eval
thread: face_identity
takeaway: >
  Base-model ID-Similarity reference on all 5 identities — the `Original` row and the pilot's hard
  gate. Gate (a)/(b)/(c) all pass for Obama/Trump/Biden/Elizabeth (Merkel misses on id_sim); the 5x5
  cross-reference matrix is built and `IDENTITY_THRESHOLD` calibrated to 0.23 (FPR 0%, TPR 78%).
  Found and fixed a silent generation-failure bug along the way: 11/150 clips had blank/structureless
  frames (see `docs/face_identity.md` §3.2); rescoring after the fix flips Biden's
  `face_present_rate` from a borderline fail (0.7884) to a clear pass (0.8448) and reframes the pilot
  pick. Pilot identities: **Obama + Queen Elizabeth II**, not Obama + Merkel — Merkel is both the
  weakest identity on every axis and the most degenerate-clip-affected (4/30), so the paper-derived
  "Obama + Merkel" guess does not survive contact with our own numbers.
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

## Results

Ran clean (`outputs_20260808_180400`, 4.61h, 150/150 videos, no exceptions), but automated scoring
initially hid a real problem: **11 of 150 clips had one or more blank/structureless frames** — 7
fully black — a silent bf16/VAE-tiling generation failure, not caught by anything (a black clip is
non-empty, so the old `getsize() > 0` resume check would have skipped it forever on any rerun).
ArcFace correctly reported "no face" on these frames, but that's a different claim from "the model
rendered no visible face" — averaging the two together understated `face_present_rate` and the
`quality` block unevenly across identities (Merkel 4/30 clips affected, Trump 4/30, Biden 2/30,
Elizabeth 1/30, Obama 0/30). Fixed in `zml/benchmarks/frame_quality.py` (pixel-intensity-std test,
calibrated on this run's own clips — see `docs/face_identity.md` §3.2) and threaded through
`check_for_face.py`/`face_eval.py` so degenerate frames are excluded from denominators the same way
a no-face frame already was. Rescored with:

```
uv run python -m zml.eval.face_eval --rescore experiments/face_identity/exp090_eval_base_face/outputs_20260808_180400 \
    --prompts-csv prompts/face_cogvideox.csv
```

**Corrected `face_present_rate` (headline gate criterion)** — face-conditioned `id_sim` barely moves
(it already excluded no-face frames from its own mean); `face_present_rate` moves visibly wherever
`clips_degenerate` is nonzero:

| | Merkel | Obama | Trump | Biden | Elizabeth |
|---|---|---|---|---|---|
| id_sim | .2734 | .5082 | .4875 | .4504 | .3272 |
| face_present_rate (before) | .6741 | .8735 | .8054 | .7884 | .8197 |
| face_present_rate (after) | **.7374** | .8735 | **.9294** | **.8448** | **.8474** |
| clips_degenerate | 4/30 | 0/30 | 4/30 | 2/30 | 1/30 |

**Gate verdict:**
- **(a)** ≥2 identities with id_sim ≥ ~0.30 — **pass**: Obama/Trump/Biden/Elizabeth all clear it;
  only Merkel misses (.2734), and this was never a degenerate-clip artifact — Merkel's face-conditioned
  score is essentially unchanged by the fix.
- **(b)** face_present_rate ≥ 0.8 on those — **pass, and stronger than the pre-fix numbers showed**:
  Biden flips from a borderline fail (.7884) to a clear pass (.8448); Trump and Elizabeth also move up.
- **(c)** 5×5 cross-reference separation matrix — **pass**, and cleanly. Built via
  `zml.eval.face_eval._cross_reference_scores` (each identity's 30 clips scored against all five
  references, 150 same-identity + 600 different-identity per-clip samples). Same-identity p25/50/75
  = .253/.379/.492; different-identity p99/p99.9/max = .108/.184/.226 — no overlap between the two
  distributions' bulk, only a thin tail. `IDENTITY_THRESHOLD` calibrated to **0.23** (just above the
  negative ceiling): FPR 0.0%, TPR 78.0% — see `docs/face_identity.md` §5 for the full table and
  `id_similarity.json`'s `cross_reference`/`cross_reference_per_clip` keys for the raw matrix.

  **Known staleness**: the cross-reference matrix was merged into `id_similarity.json` by a
  standalone script that only added the `cross_reference`/`cross_reference_per_clip` keys, without
  re-running `process_videos()` — so the file's `identity_threshold` field and `per_identity[*].identified_rate`
  still reflect the old 0.30 default, not the newly-calibrated 0.23. This affects only the
  live-training diagnostic (`face_detection_rate`), never the published `id_sim`/`face_present_rate`.
  A future `--rescore` (no extra flags needed — `IDENTITY_THRESHOLD` now defaults to 0.23 in code)
  will refresh it.

**Pilot identity pick, corrected.** The doc's own rule — highest id_sim + highest-scoring identity
that differs from it demographically — was pencilled in as Obama + Merkel off the *paper's* numbers.
Applied to ours: Obama (.5082) is the clear top; Merkel is not a viable second pick (weakest id_sim
of all five, and the most degenerate-clip-affected). Trump/Biden are the same demographic bucket as
Obama (male US politicians); **Queen Elizabeth II** (.3272, female, non-US, and now cleanly clears
both gate criteria at .8474 face_present_rate) is the corrected pick. exp092/093's identity choice
should follow this, not the frontmatter's original guess.

Visual spot-check (per `docs/face_identity.md` §3.1's discipline): a handful of clips per identity
looked clean and correctly-identified; the degenerate clips were independently confirmed blank by
eye before the fix was written (`donald_trump/video_{5,13,15,29}`, `joe_biden/video_{19,29}`,
`angela_merkel/video_29`, `queen_elizabeth_ii/video_22` fully black; `angela_merkel/video_{7,20,27}`
partially black). One further clip, `queen_elizabeth_ii/video_17`, is corrupted differently (flat
colour bands, not blank) and is not caught by the fix — a known limitation, see `docs/face_identity.md`
§3.2.

## Downstream
Picks the 2 pilot identities for exp092/093/095/096 (**Obama + Queen Elizabeth II**, corrected above)
and is the `Original` row of the final comparison table. `IDENTITY_THRESHOLD` is now calibrated
(0.23), so `face_detection_rate` is trustworthy as exp095/096's live-training signal.

## Status
- [x] `tools/fetch_face_models.py` run on the target cluster's login node.
- [x] Submitted.
- [x] Gate criteria checked (base ID-sim, face_present_rate, 5x5 separation matrix) — see Results.
- [x] `identity_threshold` calibrated and recorded here: **0.23** (FPR 0.0%, TPR 78.0%).
- [x] Pilot identities confirmed: **Obama + Queen Elizabeth II** (not Obama + Merkel — see Results).
