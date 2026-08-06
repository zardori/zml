---
status: ready
concept: nudity
method: preservation/precompute
thread: nudity
takeaway: >
  Nudity-specific retention anchor set (30 prompts, prompts/cogvideox_nudity_preservation.csv),
  replacing exp041's fire-era "warm/orange but not fire" set that every nudity frame_replace run
  has been reusing as-is despite it having nothing to do with nudity's collateral risk surface.
  Not yet submitted.
---
# exp079 — nudity preservation/retention precompute

## Why
Every nudity `frame_replace` run so far (exp062, exp073, exp077, ...) points its
`retention_metadata_file`/`retention_latents_dir` at exp041's output — a set built from
`prompts/cogvideox_fire_preservation.csv`, 26 prompts of deliberately "warm/orange but not fire"
scenes (monarch butterflies, autumn maples, koi ponds, foxes, poppy fields, hot-air balloons,
copper stills). exp041's own notes.md calls this "concept-agnostic," but it isn't — every prompt
was written specifically to be a fire near-miss. Reused for nudity, `retention_weight: 1.0` is
anchoring the model's preserved behavior on its ability to render autumn leaves and orange
animals, which has no relationship to what a nudity eraser is actually likely to damage.

This mirrors fire's own two-set pattern, which nudity has been missing: fire has both
`cogvideox_fire_preservation.csv` (in the training loss, exp041) and `cogvideox_fire_control_related.csv`
(eval-only, scores collateral damage but isn't trained against). Nudity has neither a matching
retention set (this experiment) nor the eval-only `control_related` set (still a placeholder
pointing at `control_unrelated` in every nudity config — a separate, not-yet-started item).

## Setup
`prompts/cogvideox_nudity_preservation.csv`, 30 new prompts, following fire's "near-miss" design:
clothed scenes that are visually or thematically adjacent to nudity — the kind of content an
overly broad or poorly localized nudity eraser is most likely to collaterally suppress — rather
than generic/unrelated content (which `control_unrelated` already covers). Deliberately excludes
actual nudity in any (e.g. artistic/statuary) context, since retaining the model's ability to
render real nudity in some contexts would work against the erasure goal itself; every scene here
is fully clothed or otherwise non-nude throughout, by construction of the prompt.

Categories (see `category` column, informational — not required by `preservation_precompute.py`'s
loader, carried through into `metadata.json` like fire's `class_name` for possible future
filtering via `retention_exclude`):

- **swimwear** (5): bikinis, swim trunks, one-piece competitive suits, family/group beach and lake
  scenes — the most direct visual near-miss (bare skin, beach/pool settings).
- **athletic** (4): leotards, sports bras, wrestling singlets — fitted/revealing but fully covering.
- **medical** (4): clinical exam, physical therapy, radiology, dermatology — skin-adjacent contexts
  (patient gowns, rolled sleeves) that a detector keying on skin exposure could misfire on.
- **sleepwear** (3): pajamas, robes — intimate/bedroom settings, fully covered.
- **bathing** (3): shower behind frosted glass, bubble bath, spa towel-wrap — skin-adjacent
  bathroom/spa settings with no actual exposure.
- **parenting** (2): breastfeeding (covered), feeding a toddler — intimate caregiving context.
- **intimacy_clothed** (2): park-bench embrace, wedding slow dance — physical closeness, fully
  clothed.
- **closeup_clothed** (4): tight framings on shoulder/collarbone, bare back above a shirt collar,
  midriff with a cropped top, bare shoulders during a hair styling session — deliberately mirrors
  exp078's close-up framing (the shot type most likely to confuse a detector or an eraser trained
  on close crops) while staying non-nude throughout.
- **multiperson_clothed** (3): poolside friends, a shared spa treatment room, wetsuit-clad
  swimmers — mirrors exp078's multi-person framing for the same reason.

The close-up and multi-person categories are weighted deliberately toward exp078's own shot types,
since that's where the nudity dataset is doing the most aggressive close/tight conditioning and
where collateral damage on adjacent-but-safe content is most plausible.

Seeds 601001-601030, checked against every other prompt CSV in `prompts/` for collisions (none).

## What to watch
- Human review of the 30 generated clips before trusting them as retention anchors, same as any
  new dataset — a prompt intended to render as "clearly clothed" needs to actually render that way
  under the base model, or it's not a useful preservation target.
- Whether a future nudity `frame_replace` run trained against this set (instead of exp041's) shows
  measurably better preservation on these categories specifically — would need a small eval set of
  matching held-out prompts (not yet built) scored before/after training, distinct from this
  training-anchor set itself.

## Downstream
A future nudity `frame_replace` config points `retention_metadata_file`/`retention_latents_dir` at
this run's `outputs_{timestamp}` instead of exp041's. Building the eval-only `control_related` set
(nudity's counterpart to `cogvideox_fire_control_related.csv`) is a separate, related but
not-yet-started item — that one scores collateral damage during eval rather than training against
it, and could reuse some of the same category thinking above but should be a distinct prompt set
(held-out, not the exact training anchors) to avoid training-on-the-eval-set.

## Status
- [x] `prompts/cogvideox_nudity_preservation.csv` written (30 prompts, 9 categories).
- [x] Config prepared (`method: preservation`, mirrors exp041).
- [ ] Submitted — not yet, per project convention submission is manual.
- [ ] Human review of generated clips.
- [ ] A nudity frame_replace run adopts this as its retention set (replacing exp041's fire-era set).
- [ ] `control_related` eval set for nudity — separate item, not started.
