---
status: ready
concept: face
method: frame_replace_split/precompute
thread: face_identity
takeaway: >
  Split-prompt frame_replace dataset for Queen Elizabeth II, 30 triples (seeds 7701-7730), identical
  recipe to exp092 (Obama). Confirmed as the second pilot identity by exp090's actual gate numbers
  (superseding this experiment's original Angela Merkel target — see exp090's notes.md). Not yet
  submitted.
---
# exp093 — split-prompt frame_replace dataset for Queen Elizabeth II

## Why
Second pilot identity, confirmed by exp090's actual base-model numbers to differ demographically from
Obama — non-US, female — so the split-prompt recipe (A/B/C design, de-biasing knobs) is tested for
being identity-agnostic rather than incidentally tuned to "erase a suited American man." This
experiment originally targeted Angela Merkel on the pre-run guess "highest id_sim + most
demographically distinct" applied to T2VUnlearning's *published* numbers; exp090 found Merkel is
instead the *weakest* identity on our own base model (lowest id_sim of all five, and the most
degenerate-clip-affected — 4/30, see exp090's notes.md) and does not survive contact with our
numbers. Queen Elizabeth II (next-highest id_sim after Obama, and still non-US/female) is the
corrected pick. Same rationale and construction as exp092 otherwise; see that notes.md for the full
design discussion (B must not remove the person, C must keep an unnamed person, substitute
diversity, the whole-clip hedge).

## Setup
`prompts/split_face_queen_elizabeth_ii.csv`: 30 hand-authored triples, same A/B/C design as Obama's
set but independently written scenes (garden party, stable yard, state banquet, Trooping the Colour,
opening of Parliament, ship-naming ceremony, hospital visit, Remembrance Day wreath-laying, ...) —
not the same 30 scenes with the name swapped, which would risk baking in one scene-selection bias
across both pilot identities. Same knobs as exp092 (`split_step_frac: 0.5`, `concept_region: random`,
`split_jitter: 2`, `emit_whole_clip_target: true`). Seed block 7701-7730, disjoint from Obama
(7401-7430), Merkel (7501-7530, still used by exp090/091's 5-identity baseline), the published eval
set (1065-5593), and preservation (7601-7625).

`split_step_frac` should track exp092's final decision if that run's review picks a different value
from the sweep — keep the two pilot identities on the same regime unless there's a specific reason
not to, so a difference in outcome is attributable to the identity, not the recipe.

## What to watch
Same as exp092: splice quality vs whole-clip quality (reviewed separately), B-prompt substitute
diversity, per-seed render failures, and the `tools/split_face_prompts.py` anti-cheat check — already
run locally for this CSV (`Anti-cheat check passed`) and re-enforced at generation time.

## Downstream
Feeds exp096 (`target_variant` fixed to whichever exp095 finds better) — fill its
`metadata_file`/`latents_dir` with this run's `outputs_{timestamp}`.

## Status
- [x] exp090 confirms Queen Elizabeth II as the second pilot identity (not Merkel — see Why).
- [x] `prompts/split_face_queen_elizabeth_ii.csv` authored (30 triples) and anti-cheat checked.
- [ ] Submitted.
- [ ] Dataset reviewed — splice quality and whole-clip quality, separately.
- [ ] `split_step_frac` matches exp092's final decision.
