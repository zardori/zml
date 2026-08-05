---
status: active
concept: nudity
method: frame_replace_split/precompute
thread: nudity
takeaway: >
  New 50-triple split-prompt dataset (15 close-up framings, 15 multi-person scenes, 20
  wording-diverse triples). First build (`outputs_20260804_162203`, split_step_frac=0.8, run
  before the fixes below) yielded only 25/49 kept (46.7% close-up, 33.3% multi-person, 68.4%
  wording-diverse — well under exp061's ~70% baseline), and one row (seed 3620) was silently lost
  to the now-fixed trailing-skip bug. Root cause identified 2026-08-04: the low yield was the
  NudeNet-based mask/skip logic failing, not the dataset construction — visually-inspected kept
  clips were all clean. Fixed by deriving the concept mask directly from
  (split_latent_frame, concept_region) instead of detection (see `frame_replace_split_precompute.py`
  and `docs/split_prompt.md`); the detector is now logging-only. **This dataset needs a rebuild
  with the fixed script before use** — the current kept/skipped split undercounts real yield.
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
- ~~Keep/skip yield vs exp061's ~70% baseline~~ — answered below; the yield gap was a detector
  problem, not a prompt or threshold-calibration problem.
- ~~Multi-person donor consistency~~ — answered below; this exact concern was the dominant failure
  mode, and it's now fixed structurally rather than by recalibrating a threshold.
- ~~Disappearing-last-row bug~~ — root-caused and fixed (2026-08-04, see `docs/split_prompt.md`);
  this run predates the fix and lost seed 3620.
- **Wording-diverse batch (seeds 3601-3620)**: not yet specifically re-examined post-fix; still
  worth checking once rebuilt whether the softer phrasing renders as reliably as exp061's blunter
  template, now that the mask no longer depends on detection quality either way.
- Human review pass on the kept triples before any training run, same as exp061 — still applies
  after the rebuild; construction succeeding structurally isn't the same as a scene rendering well
  (see exp074's seed-3163 finding, a persistent per-seed defect no detector-based check would catch).

## Results — first build (`outputs_20260804_162203`, split_step_frac=0.8, pre-fix script)
49/50 rows accounted for (seed 3620 lost to the trailing-skip bug). 25 kept, 24 skipped:

| batch | kept/total | yield | dominant skip reason |
|---|---:|---:|---|
| close-up (3401-3415) | 7/15 | 46.7% | `no_concept` (5/15) — detector under-firing |
| multi-person (3501-3515) | 5/15 | 33.3% | `insufficient_donor_frames` (9/15, mostly 0 donor frames) |
| wording-diverse (3601-3619) | 13/19 | 68.4% | close to exp061's ~70% baseline |

**Root cause (2026-08-04, frame-level investigation, not just the aggregate numbers):** visually
inspected 4 kept clips (2 multi-person, 2 close-up) — all four are cleanly constructed, correct
clothed→nude transitions, correct edits. The failures are in the *detector*, not the splice:
- **Flicker**: seed 3502 (multi-person, kept) — confidence drops to 0.0 for 5 frames in the middle
  of an otherwise-confidently-nude stretch (frames 38-42, sandwiched between 0.31-0.44), on a static
  scene where nothing changes. Frame-independent classifier, no temporal smoothing.
- **Under-detection on close-ups**: seed 3414 (kept, visually confirmed nude at pixel frame 44) —
  detector confidence at frame 44 is **0.0**; only one frame in the entire 49-frame clip (index 35,
  0.3235) crosses the 0.3 threshold at all. This directly explains the close-up batch's `no_concept`
  skips: NudeNet (tuned on standard full-body poses) doesn't generalize to tight anatomical crops.
- **Over-triggering on multi-person**: 9/15 multi-person skips are `insufficient_donor_frames` with
  **zero** donor frames — the detector reads the concept as present in every single latent frame,
  consistent with it not localizing per-person and instead reading "concept visible anywhere in
  frame" as concept-positive for the whole frame, exactly the risk flagged above before submission.

**Fix**: `frame_replace_split_precompute.py` now derives the concept mask directly from
`(split_latent_frame, concept_region)` — known by construction, since we choose the split point
ourselves — instead of rederiving it from NudeNet. The detector still runs and logs
`frame_confidences` for human review, but no longer gates keep/skip. See `docs/split_prompt.md` for
the full writeup. This should recover close to 100% yield (modulo genuine per-seed render failures
like exp074's seed 3163), since the two dominant failure modes above were both detector artifacts,
not construction failures.

## Downstream
Feeds a new frame_replace run (exp062-successor, not yet created) — either combined with exp061's
21 triples or as an "old vs new coverage" A/B, per the "Kept deliberately separate" note above.
Blocked on rebuilding this dataset with the fixed script first.

## Status
- [x] `split_step_frac` set to 0.8 for the first build — not blocking on exp076, see Setup.
- [x] Submitted and run (`outputs_20260804_162203`).
- [x] Kept/skipped counts recorded; yield compared against exp061's ~70% baseline — see Results.
- [x] Human review of a sample of kept triples (close-up crop quality, multi-person donor edit
      sanity) — both looked clean; see Results.
- [x] Root cause of the low yield identified and fixed (detector-driven masking, not the prompts or
      the splice) — see Results and `docs/split_prompt.md`.
- [ ] **Rebuild with the fixed script** (and `split_step_frac: 0.85`, the current default) to get
      the real yield — not yet done.
- [ ] Next frame_replace run's dataset composition decided (old+new vs new-only vs A/B).
