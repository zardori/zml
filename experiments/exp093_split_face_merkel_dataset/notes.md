---
status: ready
concept: face
method: frame_replace_split/precompute
thread: face_identity
takeaway: >
  Split-prompt frame_replace dataset for Angela Merkel, 30 triples (seeds 7501-7530), identical
  recipe to exp092 (Obama). Blocked on exp090's pilot-identity confirmation. Not yet submitted.
---
# exp093 — split-prompt frame_replace dataset for Angela Merkel

## Why
Second pilot identity, chosen (pending exp090's actual numbers) to differ demographically from Obama
— non-US, female — so the split-prompt recipe (A/B/C design, de-biasing knobs) is tested for being
identity-agnostic rather than incidentally tuned to "erase a suited American man." Same rationale and
construction as exp092; see that notes.md for the full design discussion (B must not remove the
person, C must keep an unnamed person, substitute diversity, the whole-clip hedge).

## Setup
`prompts/split_face_angela_merkel.csv`: 30 hand-authored triples, same A/B/C design as Obama's set
but independently written scenes (parliament chamber, chemistry lab reflecting her PhD background,
factory tour, opera box, ...) — not the same 30 scenes with the name swapped, which would risk
baking in one scene-selection bias across both pilot identities. Same knobs as exp092
(`split_step_frac: 0.5`, `concept_region: random`, `split_jitter: 2`, `emit_whole_clip_target: true`).

`split_step_frac` should track exp092's final decision if that run's review picks a different value
from the sweep — keep the two pilot identities on the same regime unless there's a specific reason
not to, so a difference in outcome is attributable to the identity, not the recipe.

## What to watch
Same as exp092: splice quality vs whole-clip quality (reviewed separately), B-prompt substitute
diversity, per-seed render failures, and the `tools/split_face_prompts.py` anti-cheat check (already
enforced at generation time for this CSV too).

## Downstream
Feeds exp096 (`target_variant` fixed to whichever exp095 finds better) — fill its
`metadata_file`/`latents_dir` with this run's `outputs_{timestamp}`.

## Status
- [ ] exp090 confirms Merkel as a pilot identity (expected, pending that run).
- [ ] Submitted.
- [ ] Dataset reviewed — splice quality and whole-clip quality, separately.
- [ ] `split_step_frac` matches exp092's final decision.
