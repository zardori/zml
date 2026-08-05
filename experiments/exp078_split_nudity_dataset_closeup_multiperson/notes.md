---
status: ready
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
  and `docs/split_prompt.md`); the detector is now logging-only. Also added `boundary_margin`
  (default 2): since the concept block always touches a clip edge here, `edit_latent` always copies
  the single nearest safe frame across the whole block rather than interpolating — that one frame's
  cleanliness matters a lot, and margin pulls it further from the boundary (which the heal phase's
  joint cross-attention could otherwise have touched). Then corrected further: margin alone only
  changes *which* frame gets frozen, not the freeze itself — `edit_latent`'s own docstring warns a
  hard single-frame copy suppresses motion (exp055: concept -84%, unrelated -29%), and this
  construction hits that fallback on every clip. Replaced with `edit_latent_reflected` — mirrors
  the safe segment's motion into the concept region (bouncing back and forth if needed) instead of
  freezing one frame. **This dataset needs a rebuild with the fixed script before use** — the
  current kept/skipped split undercounts real yield, and the kept clips' edited halves are frozen
  single-frame copies rather than motion-preserving fills. **Rebuild is now a `split_step_frac`
  grid `[0.8, 0.85, 0.9, 0.95, 1.0]` on this dataset's actual 50 triples**, not a single value —
  exp074/076's sweep that picked 0.85 never specifically checked for boundary concept-mixing on
  close-up/multi-person framing, which is exactly what turned out to be faulty here. A higher
  split_step_frac leaves less schedule for the heal phase's joint cross-attention, so directly
  reduces the opportunity for that mixing — worth re-checking on the framings that actually need it
  rather than trusting the generic sweep's answer. Not yet submitted.
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
only.

**`split_step_frac` (rebuild, 2026-08-04): now a grid `[0.8, 0.85, 0.9, 0.95, 1.0]`, not a fixed
value.** The first build used the single best-confirmed value from exp074/076's sweep (0.85), but
that sweep only tested 5 simple, single-person, full-body seeds (`prompts/split_nudity_sweep.csv`)
and never specifically checked for concept mixing across the split boundary — which turned out to
be a real problem in this dataset's construction (see `boundary_margin`/`edit_latent_reflected`
above). `split_step_frac` is a direct lever on that same mechanism: it's the fraction of the
schedule spent in the temporally-split A/B phase before the shared heal phase (conditioned jointly
on prompt C, with full cross-attention over the whole clip) takes over — a higher value leaves less
of the schedule for that joint phase, so less opportunity for it to blend information across the
boundary. Sweeping on this dataset's actual 50 triples (close-up/multi-person, the framings that
actually surfaced the mixing problem) checks the real thing this dataset needs, rather than
assuming the generic sweep's answer (0.85) transfers. `1.0` (zero heal-phase steps, zero
opportunity for cross-boundary mixing) is included even though exp076 didn't pick it for the
generic sweep — its reasoning ("no measured benefit") doesn't hold if mixing is a real cost that
generic sweep never measured.

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
- [x] Config updated to a `split_step_frac: [0.8, 0.85, 0.9, 0.95, 1.0]` grid rebuild with the
      fixed script (mask from construction, `boundary_margin`, `edit_latent_reflected`) — see
      Setup. **Not yet submitted** — per project convention, submission is manual.
- [ ] Submit and run the grid (5 jobs, ~9h budget each).
- [ ] Compare kept/skipped yield (should be ~100% across all 5 values now) and, more importantly,
      check for boundary concept-mixing per value — frame-level review at the seam, not just the
      aggregate detector score, per [[feedback-detector-metrics-not-ground-truth]].
- [ ] Pick a final `split_step_frac` for this dataset's real build (may differ from exp076's 0.85).
- [ ] Human review of kept triples on the winning value.
- [ ] Next frame_replace run's dataset composition decided (old+new vs new-only vs A/B).
