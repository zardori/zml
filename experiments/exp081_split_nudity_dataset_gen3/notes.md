---
status: ready
concept: nudity
method: frame_replace_split/precompute
thread: nudity
takeaway: >
  Third-generation split-prompt nudity dataset, 55 new hand-written triples
  (prompts/split_nudity_gen3.csv, seeds 3701-3755), each its own sentence rather than one
  noun-substitution template (a first draft doing the latter was correctly rejected as lazy).
  Weighted toward multi-person and creative full-body settings, exp078's best-performing
  categories; close-up narrowed to a looser two-region "medium close" crop instead of exp078's
  isolated tight crop, testing whether crop tightness was the actual failure mode. Not yet
  submitted — split_step_frac still pending exp078's run_001-004 review.
---

# exp081 — split-prompt nudity dataset, gen3

## Why
Running tally of human-confirmed-usable nudity triples: exp061's 21 (single-person, plain
studio/clinical settings, one rigid template) + exp078 run_005's 13 (`split_step_frac: 1.0`,
`metadata_human_filtered.json`, now also consumed by exp080's training run) = **34** approved so
far. Target is ~100. Rather than repeat exp078's even 3-way split, this batch acts on what exp078's
review actually showed:

| exp078 batch | approved/total | rate |
|---|---:|---:|
| multi-person | 6/15 | 40.0% |
| wording-diverse | 5/20 | 25.0% |
| close-up | 2/15 | 13.3% |

(Numbers are from run_005 only — the other 4 grid values were still under review as of
2026-08-06, so absolute rates may shift, but the *relative* ordering across batches — all 5 grid
runs share the same 50 underlying prompts, differing only in `split_step_frac` — is the most
robust signal available right now.)

**First draft of this dataset was rejected as lazy** (2026-08-06): generated 70 triples from one
sentence skeleton ("Static [shot] of [comp] ... in [setting], [nude phrase], full anatomical
detail, photorealistic. The camera is fixed and never moves.") with only the nouns swapped from
list pools via a Python script. This is close to the exact failure mode exp078's own
wording-diverse batch was built to address — the model latching onto exact repeated training
phrasing instead of the underlying concept — and memory
([[project-nudity-splitprompt-dataset]]) already flags this risk explicitly. Rewritten from
scratch as 55 hand-authored sentences instead; smaller than the original 70-row target but every
row has genuinely distinct sentence construction, which matters more for this specific
generalization concern than raw count.

## Setup
`prompts/split_nudity_gen3.csv`, 55 new A/B/C triples. No shared sentence skeleton — clause order,
descriptive style, and camera-language phrasing vary row to row; only the semantic requirement of a
static/fixed camera (needed for the split sampler's temporal splice) repeats, worded differently
almost every time ("the camera never moves" / "static, unmoving shot" / "the shot holds perfectly
still" / "camera fixed and never moves" / "the camera does not move" / ...).

- **3701-3723 (23, multi-person):** scaled up from exp078's 15, leaning into the "plausible
  practical/unusual scenario" register that seemed to be the common thread behind exp078's
  multi-person wins (clinical exam room, life-drawing studio, wine cellar, ceramics workshop) —
  extended with bathhouse, decompression chamber, greenhouse at midnight, submarine berth,
  blacksmith's forge, glassblowing kiln, ice hotel, daguerreotype-era portrait studio, apiary
  washstand, mountain-spring bathing, tattoo studio, backstage dressing room, boathouse dock, opera
  backstage, barn trough, museum vault, yoga studio, among others. Compositions vary (couples,
  trios, family groups, colleagues) and relationships (embracing, working side by side, waiting,
  rehearsing) rather than a fixed "standing/seated" pool.
- **3724-3744 (21, single-person, full-body/medium):** same framing register as exp061 (already
  confirmed to work), scaled into more varied and unusual-but-plausible settings for coverage:
  lighthouse tower, recording booth, conservatory koi pond, observatory dome, mine tunnel,
  planetarium booth, ice cave, barn tightrope practice, wine cellar, bank vault, houseboat deck,
  cinema projection booth, vintage train car, falconer's mews, greenhouse potting shed,
  clockmaker's workbench, abandoned opera box, apiary, submarine engine room, sculptor's studio,
  frozen lake.
- **3745-3755 (11, medium-close, deliberately looser than exp078's close-up):** two-region crops
  (shoulders+upper back, torso+hips, chest+stomach, hips+lower back, back+shoulder blades,
  chest+collarbone, hips+thighs) instead of exp078's single isolated region, testing whether
  exp078's 13.3% close-up approval rate came from crop tightness specifically (harder for the model
  to render convincingly, or harder for the mask/splice to land cleanly at that scale) rather than
  close framing being inherently bad. Kept smaller than the other two batches until this theory is
  validated by review.

Split sampler knobs (`split_latent_frame: 7`, `concept_region: random`, `split_jitter: 2`,
`frame_nudity_threshold: 0.3`, `boundary_margin: 2`) carried over unchanged from exp078 — same
construction (mask-from-construction, `edit_latent_reflected` fill), new prompts only.

Kept as its own CSV/experiment, not merged into `split_nudity.csv` or `split_nudity_gen2.csv`, for
the same reason exp078 stayed separate from exp061: lets old/gen2/gen3 be compared or combined
deliberately later (`zml/precompute/merge_frame_replace_datasets.py`, used by exp080, is the
existing tool for this) rather than forcing a merge before any of them has been reviewed.

## What to watch
- Whether the multi-person-weighted, sentence-diverse mix raises the overall approval rate over
  exp078's ~46% aggregate (34/74 submitted-and-reviewed so far across all batches) — the real test
  of both this batch's bets (lean into what worked; write real sentences instead of templates).
- The medium-close batch (3745-3755) specifically: does loosening the crop from exp078's
  single-region tight crop to a two-region "medium close" recover approval rate toward the
  multi-person/single-person range?
- Standard split-prompt failure modes carried over from exp061/74/78: per-seed render failures no
  detector-based check catches (see exp074's seed-3163 finding) — still needs a human video review
  pass on the kept triples before any of this feeds a training run, same as every prior batch.

## Status
- [x] `prompts/split_nudity_gen3.csv` generated — 55 hand-written triples, seeds 3701-3755.
- [x] `config.yaml` written, `job_type: precompute`, `method: frame_replace_split`.
- [ ] `split_step_frac` finalized — blocked on exp078's run_001-004 human review. Placeholder value
      (0.85) in config must not be trusted for submission until that review lands.
- [ ] Submitted (manual, per project convention — not done by Claude).
- [ ] Human review of kept triples.
