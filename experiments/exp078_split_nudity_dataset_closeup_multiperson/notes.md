---
status: ready
concept: nudity
method: frame_replace_split/precompute
thread: nudity
takeaway: >
  New 50-triple split-prompt dataset (15 close-up framings, 15 multi-person scenes, 20
  wording-diverse triples) targeting two coverage gaps exp062's human review + follow-up
  discussion diagnosed: the existing 21-triple dataset is all single-person/fully-visible/static
  with one repeated prompt template, while the eval set (cogvideox_nudity.csv) has close-ups,
  multiple people, AND varied narrative vocabulary. Not yet submitted.
---
# exp078 — split-prompt nudity dataset, close-up + multi-person coverage

## Why
exp062 run 3 looked like clean erasure by `nudity_detection_rate` alone, but human video review
(see exp062's notes.md, 2026-08-04) found only partial/inconsistent unlearning — and specifically
pointed at dataset narrowness as the cause: every one of exp061's 21 confirmed-good triples is the
same shot type (`prompts/split_nudity.csv`: "Static [wide/medium] shot of a completely naked
**person** standing/seated **upright**..., full anatomical detail... camera fixed and never
moves"). Compare that to `prompts/cogvideox_nudity.csv` (the actual eval set, i2p-derived) — full of
close-up shots ("close-up shot... focusing on her face", "the camera zooms in on their naked
bodies") and multi-person scenes ("A couple, naked and embracing...", "Two individuals..."). The
training data was never teaching the model to erase nudity in the framings the eval set actually
tests. This dataset targets exactly that gap.

## Setup
`prompts/split_nudity_gen2.csv`, 50 new A/B/C triples. Static camera held fixed throughout every
row (required for the split-prompt sampler's temporal-region splice to make sense — the schedule
needs one stable scene, not camera motion, across the A/B/C conditioning switch):

- **15 close-up triples** (seeds 3401-3415): tight framings on torso/chest, back, hips, shoulders,
  midsection — cropped so the whole body is never in frame, varying which region is cropped and the
  room setting (studio, exam room, bedroom, sauna, gym locker room, etc., reusing exp061's setting
  variety). Uses exp061's rigid prompt template ("Static [shot] of a completely naked/bare
  [region]... full anatomical detail, photorealistic...").
- **15 multi-person triples** (seeds 3501-3515): two (occasionally three) people in frame together,
  varying composition (side by side, one standing/one seated, one kneeling/one standing) and
  setting. Same rigid template as the close-up batch.
- **20 wording-diverse triples** (seeds 3601-3620, added after a follow-up discussion on generalization
  gaps): exp061/the first two batches above all share one prompt template and vocabulary
  ("completely naked... full anatomical detail, photorealistic..."), while the eval set
  (`cogvideox_nudity.csv`, i2p-derived) uses much more varied narrative phrasing ("her body
  glistening", "bare skin", "fully exposed", "unclothed", "in the nude"). If the model latches onto
  the training set's exact phrasing rather than the underlying concept, that's a second
  generalization gap layered on top of the framing one — this batch targets it directly with 5
  distinct sentence templates and varied vocabulary ("entirely bare", "completely undressed",
  "without a stitch of clothing", "nude figure", "unclothed body"), mixed across full-body (10),
  close-up (5), and multi-person (5) framing so it stacks with the first two batches rather than
  re-narrowing to one framing type.

Each prompt_b substitutes a distinct garment/coverage description (parka, robe, cloak, ski suit,
trench coat, ...) matching exp061/exp066's practice of varying the concept-free substitute rather
than reusing one fixed replacement — including in the wording-diverse batch, where the garment
description also varies in phrasing, not just item. prompt_c drops all clothing/nudity language,
matching the framing/setting only (and, in the wording-diverse batch, echoes that row's sentence
structure minus the clothing state, so C isn't a giveaway of which state A/B differ on).

Split sampler knobs (`split_latent_frame: 7`, `concept_region: random`, `split_jitter: 2`) and
`frame_nudity_threshold: 0.3` carried over unchanged from exp061 — same construction, new prompts
only. `split_step_frac: 0.8` is the best-confirmed value as of this run: exp074's human review found
0.4-0.8 all consistently good with an upward tendency and no confirmed ceiling, and exp076 (running
in parallel) is testing 0.85-1.0 to find where it turns over. Submitting on 0.8 now rather than
waiting for exp076 — compute isn't the constraint, so sequencing them only burns calendar time; if
exp076 finds something better, rebuilding this dataset with the new value is cheap.

Kept deliberately **separate** from exp061's 21-triple dataset (not merged into
`prompts/split_nudity.csv`) so a future frame_replace run can compare "old only" vs "old + new" vs
"new only" cleanly, and so a bad generation batch here doesn't put exp061's already
human-confirmed-good triples back into question.

## What to watch
- **Keep/skip yield** (`no_concept` / `insufficient_donor_frames` in `skipped.json`). Close-up
  framings may behave differently than the wide/medium shots the 0.3 threshold was calibrated on —
  a much lower yield than exp061's ~70% auto-keep rate would be a sign `frame_nudity_threshold`
  needs recalibrating for tight crops, not that the prompts are bad.
- **Multi-person donor consistency**: `edit_latent`'s interpolation assumes a single concept region
  per clip; with two people in frame, check whether the detector's frame-level mask (not
  per-person) produces a sensible donor edit rather than one person's concept frames being masked
  out while the other's aren't.
- Same disappearing-last-row(s) bug seen on `split_nudity.csv` (rows 29, 51) and on the chain-saw/
  church CSVs — check `metadata.json` + `skipped.json` account for all 50 rows before trusting the
  keep count.
- **Wording-diverse batch (seeds 3601-3620) specifically**: these prompts describe nudity/coverage
  less explicitly/clinically than exp061's template ("entirely bare", "nothing covering the skin")
  — check this doesn't reduce the splice's reliability (weaker A/B conditioning contrast than the
  blunter "completely naked... full anatomical detail" phrasing) before assuming a lower yield here
  means the threshold is miscalibrated rather than the wording being too soft to render clearly.
- Human review pass on the kept triples before any training run, same as exp061 — auto-kept
  (`no_concept`/`insufficient_donor_frames` cleared) is not the same as visually good.

## Downstream
Feeds a new frame_replace run (exp062-successor, not yet created) — either combined with exp061's
21 triples or as an "old vs new coverage" A/B, per the "Kept deliberately separate" note above.

## Status
- [x] `split_step_frac` set to 0.8 (best-confirmed so far) — not blocking on exp076, see Setup.
- [ ] Submitted.
- [ ] Kept/skipped counts recorded; yield compared against exp061's ~70% auto-keep rate.
- [ ] Human review of kept triples (close-up crop quality, multi-person donor edit sanity).
- [ ] Next frame_replace run's dataset composition decided (old+new vs new-only vs A/B).
