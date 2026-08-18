---
status: done
concept: face
method: frame_replace_split/precompute
thread: face_identity
takeaway: >
  BUILT 30/30, SCREENS 0/30 — exp096 is blocked. Every row fails the Obama-calibrated 0.30 gate on
  the original clip, and the misses are narrow: peak orig_max is 0.302 / 0.299 / 0.282 / 0.280, i.e.
  clustered just under the threshold rather than absent. That tracks exp090 exactly, where Elizabeth's
  base id_sim is 0.3272 against Obama's 0.5081 — the base model renders her much more weakly, so a
  gate tuned on Obama rejects her whole set. Two readings, and the thread owner has to pick: the
  threshold is identity-relative and needs rescaling per identity, or Elizabeth is too weak a target
  to erase and the second pilot identity should change. Note the whole-clip A gate does clear 0.30 on
  6 rows (up to 0.444), so the prompts do sometimes render her — it is the split that loses her.
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
`prompts/face_identities/split/queen_elizabeth_ii.csv`: 30 hand-authored triples, same A/B/C design as Obama's
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

## Results (2026-08-12) — built clean, screens to nothing

Built **30/30 with zero skips** (3.3 h on helios, `outputs_20260811_185219`). The build is fine; the
screen is the problem.

`tools/screen_split_face_dataset.py` at its defaults (`MIN_ORIGINAL_MAX_CONFIDENCE` 0.30,
`MIN_WHOLECLIP_A_MAX_CONFIDENCE` 0.30) keeps **0 of 30**.

| | best rows |
|---|---|
| `orig_max` (identity in the original clip) | 0.302, 0.299, 0.282, 0.280, 0.279, 0.256 |
| `wc_a_max` (identity in the whole-clip A target) | 0.444, 0.439, 0.429, 0.363, 0.232, 0.235 |

Nine rows read exactly 0.000 on `orig_max`, but the top of the distribution sits *at* the gate rather
than far below it — this is a near-miss set, not an empty one.

### Why: the gate is calibrated on Obama

exp090's base-model numbers make this predictable in hindsight:

| identity | base id_sim | base identified_rate |
|---|---|---|
| Barack Obama | 0.5081 | 0.8667 |
| Queen Elizabeth II | 0.3272 | 0.6000 |

CogVideoX renders Elizabeth at roughly two-thirds of Obama's identity strength, and `IDENTITY_THRESHOLD`
itself is 0.23 — so a 0.30 screening gate sits *above* her base-model average. exp115/exp116's
keep-lists were selected with these same defaults on Obama, where 0.30 is comfortably below his 0.508.

### The fork this leaves

1. **Rescale the gate per identity** (e.g. as a fraction of that identity's exp090 base id_sim). Then
   Elizabeth's 0.28-0.30 rows are legitimate positives and this dataset is usable, and exp115/exp116's
   Obama yields should be re-derived under the same rule for consistency.
2. **Change the second pilot identity.** exp090 ranked Trump (0.4876) and Biden (0.4504) well above
   Elizabeth; either would give a second identity the base model actually renders. This costs a new
   CSV and one precompute job, and makes the pilot's two columns comparable in a way Obama/Elizabeth
   are not.

The `wc_a_max` column is the evidence that this is a threshold/target question rather than a broken
build: six rows clear 0.30 on the whole-clip A target (up to 0.444), so the prompts *can* render her.

**This is not the same failure exp116 diagnosed for Obama.** There the problem was framing and the fix
was prompt rewriting (30% -> 50/63%). Here the prompts render her about as well as the base model ever
does; the ceiling is the model's grasp of the identity.

## Status
- [x] exp090 confirms Queen Elizabeth II as the second pilot identity (not Merkel — see Why).
- [x] `prompts/face_identities/split/queen_elizabeth_ii.csv` authored (30 triples) and anti-cheat checked.
- [x] Submitted and complete (helios, 3.3 h, 30/30 built, 0 skipped).
- [x] Screened: **0/30 at the default gates.**
- [ ] **Decide the fork above** — rescale the gate per identity, or swap the second pilot identity.
      exp096 and exp098 are blocked until this is settled.
- [ ] Dataset reviewed by eye — splice quality and whole-clip quality, separately. Worth doing on the
      six `wc_a_max` > 0.30 rows before concluding the set is unusable.
