---
status: ready
concept: face
method: frame_replace_split/precompute
thread: face_identity
takeaway: >
  Rebuild of exp092 with split_step_frac raised 0.5 -> 0.8: human review of exp092 found the
  originals look good but the merged/wholeclip target is clearly over-merged (one face carrying
  both Obama's and the substitute's features), so more of the schedule is pushed into the shared
  heal phase, closer to nudity's settled 0.85. Not yet submitted.
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
Identical to `exp092_split_face_obama_dataset/config.yaml` (same `prompts/split_face_barack_obama.csv`,
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

## Status
- [ ] Submitted.
- [ ] Dataset reviewed — splice quality and whole-clip quality, separately.
- [ ] Compare wholeclip identity separation against exp092's 0.5 build.
