---
status: done
concept: face
method: frame_replace_split/precompute
thread: face_identity
takeaway: >
  Rebuild of exp092 with split_step_frac raised 0.5 -> 0.8: human review of exp092 found the
  originals look good but the merged/wholeclip target is clearly over-merged (one face carrying
  both Obama's and the substitute's features), so more of the schedule is pushed into the shared
  heal phase, closer to nudity's settled 0.85. DONE: 0.8 fixes the chimera-face problem, but yield
  is low — 9/30 (30%) pass both splice and whole-clip identity separation, against exp092's 0.5
  build. Filtered set (seeds 7406,7407,7410,7413,7417,7421,7425,7427,7428) at
  `metadata_human_filtered.json`.
---
# exp115 — split-prompt frame_replace dataset for Barack Obama, split_step_frac 0.8

## Why
exp092's human review (2026-08-12) found a clean split between the two halves of the output:
`*_original.mp4`/`*_edited.mp4` look good, but the whole-clip target variant does not — the
merged/blended clip shows a single face carrying features of **both** Obama and the anonymous B
substitute, i.e. an identity chimera, rather than two distinct people. This is the seam-blending
failure mode exp092/notes.md already flagged as a risk ("a chimera face at the seam ... blending two
people"), just observed directly instead of hypothetically.

exp092 used `split_step_frac: 0.5`, deliberately lower than nudity's settled 0.85 on the reasoning
that identity needed a longer heal phase to hide the seam — but 0.5 apparently isn't enough schedule
in the shared C-conditioned phase to fully separate the two identities. This run tries `0.8`, close
to nudity's value, as the next point rather than committing to a full sweep yet.

## Setup
Identical to `exp092_split_face_obama_dataset/config.yaml` (same `prompts/face_identities/split/barack_obama.csv`,
30 triples, seeds 7401-7430; same `split_latent_frame: 7`, `concept_region: random`,
`split_jitter: 2`; same `emit_whole_clip_target: true`) with only `split_step_frac: 0.5 -> 0.8`
changed.

## What to watch
Same as exp092's "What to watch" section — splice quality (`*_original.mp4` vs `*_edited.mp4`) and
whole-clip quality (`*_wholeclip_a.mp4` vs `*_wholeclip_b.mp4`) reviewed separately — plus
specifically: does `wholeclip_b` now read as a single coherent (non-Obama) person rather than a
blend of the two identities.

## Downstream
If review confirms 0.8 fixes the chimera-face problem, this (not exp092) should feed exp095's
`target_variant` grid. If 0.8 still shows blending, narrow the sweep between 0.5 and 0.85/1.0 rather
than jumping further.

## Results (2026-08-13) — 9/30 kept (30%)

Reviewed all 30 triples clip by clip, splice quality and whole-clip identity separation together.
Kept: p5, p6, p9, p12, p16, p20, p24, p26, p27 (seeds 7406, 7407, 7410, 7413, 7417, 7421, 7425, 7427,
7428). Filtered metadata at `metadata_human_filtered.json` (experiment root, git-tracked), written
with `tools/filter_retention_metadata.py --allow-skew` — the `MIN_OVERALL_KEEP_FRACTION` guard is
calibrated for retention sets (exp104's 97.5%), not split-prompt triples, which fail far more often
by construction; see exp109's note on the same override.

`split_step_frac: 0.8` does fix the chimera-face failure mode exp092 flagged at 0.5 — no surviving
triple shows the blended-identity artifact in `*_wholeclip_b.mp4`. The cost is yield: 30% here
against exp092's build (0.5), which had good splices throughout but was unusable on the whole-clip
side. Not a like-for-like comparison since exp092 was never filtered end-to-end (it was superseded
before a keep list was written), but the direction is as expected — more schedule pushed into the
shared heal phase heals the seam better and generalizes worse per-triple.

## Status
- [x] Submitted.
- [x] Dataset reviewed — splice quality and whole-clip quality, separately (see Results above).
- [x] Compare wholeclip identity separation against exp092's 0.5 build — 0.8 fixes the chimera-face
      problem; exp092 is superseded (see its notes.md).

9/30 is too thin for exp095's training run on its own. `exp116_split_face_obama_dataset_scaleup`
scales this dataset up (re-seed + framing-controlled prompts, same `split_step_frac: 0.8`) and
merges its keeps with this experiment's 9 into a combined dataset for exp095. These 9 triples are
`src0_*` inside exp116's `combined_dataset/` once `./merge_dataset.sh` builds it.
