---
status: done
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

## Results (`outputs_20260806_194011`) — built clean, and it already proved its own point

30/30 latents built, all 9 categories intact, no failures.

**Verified the anchors are actually clothed (2026-08-07).** `preservation_precompute.py` runs no
detector, so nothing would otherwise catch an anchor that accidentally rendered nudity — which
would be actively harmful, since training *toward* such an anchor teaches the model to keep
generating the thing we are erasing. Ran NudeNet over the 30 sanity videos locally: **13/30 trip
the detector at frame level** (max confidence >= 0.3), topped by seed 601001 (red bikini, **0.844**,
all 49/49 frames), 601026 (midriff, 0.730), 601003 (one-piece swimsuit, 0.711), 601008 (sports bra,
0.691).

Visual check of all four worst offenders: **every one rendered exactly as written and is fully
clothed** — a red bikini on a beach, a one-piece competitive swimsuit on a starting block, a navy
sports bra and shorts, a light-blue sports bra on a treadmill. Nothing to remove; the prompts did
their job. NudeNet is simply firing on exposed skin.

So the set is correct, and it is *useful precisely because* the detector behaves this way: these are
the scenes an over-broad eraser destroys. It is also the cleanest evidence we have for
[[feedback-detector-metrics-not-ground-truth]] — a paper-quotable demonstration that NudeNet reports
0.844 on a woman who is wearing a swimsuit.

**Design tension this surfaces, worth deciding before training on it.** The retention loss pulls the
model *toward* these anchors while the erase loss pushes nudity *down*, and on this content the two
objectives disagree by construction — the same eval detector that scores our erasure calls the
bikini anchor 0.844. Two consequences:
1. A model that correctly preserves swimwear can be scored as "still generating nudity" by
   `nudity_detection_rate`, and conversely a low detection rate may mean we destroyed swimwear —
   a failure, not a success. This is the strongest argument yet for reporting a `related` column
   rather than the concept column alone.
2. The most extreme anchors (601001 at 0.844 across every frame, 601003, 601008, 601026) are worth
   a deliberate keep/drop decision rather than being included by default: they are the ones most
   likely to fight the erase objective. Recommend keeping them (they are legitimate content we
   must not destroy) but reporting them as their own slice, so preservation on skin-heavy content
   is visible separately from preservation in general.

## Status
- [x] `prompts/cogvideox_nudity_preservation.csv` written (30 prompts, 9 categories).
- [x] Config prepared (`method: preservation`, mirrors exp041).
- [x] Submitted and built — 30/30 latents, `outputs_20260806_194011`.
- [x] Anchors verified genuinely clothed (NudeNet pass + visual check of the worst 4) — see above.
- [ ] Decide the keep/drop question on the highest-scoring skin-heavy anchors (see tension above).
- [ ] A nudity frame_replace run adopts this as its retention set (replacing exp041's fire-era set).
- [ ] `control_related` eval set for nudity — separate item, not started.
