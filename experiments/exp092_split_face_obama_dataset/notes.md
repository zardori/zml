---
status: superseded
concept: face
method: frame_replace_split/precompute
thread: face_identity
takeaway: >
  Split-prompt frame_replace dataset for Barack Obama, 30 triples (seeds 7401-7430), split_step_frac
  0.5. Original/edited splices look good, but the whole-clip target is over-merged (one chimera face
  blending Obama and the substitute) -- 0.5 under-heals. Superseded by exp115 (split_step_frac 0.8).
---
# exp092 — split-prompt frame_replace dataset for Barack Obama

## Why
First frame_replace training dataset for the face-identity axis. A named identity's face is present
for the whole clip — exactly the nudity problem, but sharper: identity is maximally salient in every
frame, not just some regions of some frames, so the partiality has to be manufactured
(`docs/split_prompt.md`) and the seam is the most visually obvious this method has had to hide yet.

## Setup
`prompts/split_face_barack_obama.csv`: 30 hand-authored A/B/C triples (not a single sentence
skeleton with noun-substitution — the exact failure mode exp081's first draft was rejected for).
**A** names Obama in a plain scene; **B** is the same scene and role with an anonymous person whose
description (age, hair, build, clothing) varies row to row — never a fixed substitute (that would
train one specific face swap rather than removal, see R6 below), never another real person, never
one of the other four protocol identities; **C** keeps a person in the same scene, unnamed, camera
language only. **The sharpest deviation from the object/nudity recipe**: unlike those, B must not
*remove* the person (that teaches "delete humans," exactly the collateral Preserve measures), and C
must *keep* a person too (the heal phase conditions the whole latent on C — dropping the person there
would push the face out of the concept half as well).

`concept: face` + `concept_target: "Barack Obama"` selects the ArcFace detector
(`zml/benchmarks/check_for_face.py`) via the registry; it runs for `frame_confidences` logging only,
never gates keep/skip (the mask is known from `split_latent_frame`/`concept_region` by construction,
same as nudity post-exp078-fix).

`emit_whole_clip_target: true` additionally generates plain clips for prompt A and prompt B with the
same seed, saved under `variants.wholeclip` in `metadata.json` alongside the flat
`variants.split` keys — `zml.unlearn.unlearn_frame_replace.Config.target_variant` selects which a
training run consumes. This is the hedge against R4 (the splice may simply fail for a concept this
seam-visible): if human review finds the spliced clips unusable, the whole-clip target still gives a
usable dataset from the same GPU pass, no second precompute run needed.

`split_step_frac: 0.5`, lower than nudity's settled 0.85 — more schedule in the shared heal phase,
since identity has the least tolerance for a visible seam of any concept tried so far. Not yet swept;
revisit (0.4/0.5/0.6) if this build's human review finds a bad yield.

## What to watch
- **Splice quality.** Visually review `videos/*_original.mp4` vs `*_edited.mp4` per triple. Two
  failure modes specific to identity: a chimera face at the seam (the heal phase blending two
  people), and the reflected fill making the identity region look like a different, blended person
  rather than a clean cut.
- **Whole-clip quality**, separately: `videos/*_wholeclip_a.mp4` (should clearly be Obama) vs
  `*_wholeclip_b.mp4` (should clearly not be — and should render an equally plausible anonymous
  person in the same scene, not an empty one).
- **B-prompt substitute diversity.** The 30 B descriptions should read as genuinely different people,
  not one archetype with swapped clothing — the concrete case R6 (`docs/face_identity.md`) worries
  about is the LoRA learning a *specific* replacement face rather than removal in general.
- Standard split-prompt failure modes: per-seed render failures no detector-based check catches
  (exp074's seed-3163 finding, nudity's analogue).
- **Anti-cheat, already enforced by `tools/split_face_prompts.py`**: none of these 30 `prompt_a`
  rows appear in `prompts/face_cogvideox.csv` (verified at generation time; rerun the tool if this
  CSV is ever edited).

## Downstream
Was intended to feed exp095 (`target_variant: [split, wholeclip]` grid), but the over-merged
whole-clip target makes this build unsuitable for the `wholeclip` arm — see Status below. exp095
should instead wait on [[exp115]]'s `split_step_frac: 0.8` rebuild.

## Status
- [x] exp090 confirms Obama as a pilot identity (highest base-model id_sim of all five).
- [x] Submitted.
- [x] Dataset reviewed — splice quality and whole-clip quality, separately (see What to watch).
  **Result (2026-08-12):** `*_original.mp4`/`*_edited.mp4` look good, but the whole-clip target is
  clearly over-merged — `*_wholeclip_b.mp4` shows a single chimera face carrying features of both
  Obama and the B-prompt substitute, rather than two distinct people. `split_step_frac: 0.5`
  apparently doesn't give the heal phase enough schedule to fully separate the two identities.
- [x] `split_step_frac` sweep decided: 0.5 under-heals. Trying `0.8` (closer to nudity's settled
  0.85) next in [[exp115]] rather than the originally planned 0.4/0.5/0.6 sweep.
