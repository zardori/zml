# Split-Prompt: Manufacturing Partial-Concept Clips

This document describes how frame_replace training data is built for concepts that are **not
naturally partial**. It is the reference behind `zml/precompute/split_prompt_precompute.py`
(method `split_prompt`, the sampler) and `zml/precompute/frame_replace_split_precompute.py`
(method `frame_replace_split`, the full dataset builder). For the unlearning method that consumes
these targets, see [`frame_replace.md`](frame_replace.md).

---

## 1. The transfer problem

frame_replace needs **partial-concept clips**: the concept must be present in some frames of a clip
and absent in others, so the absent frames can serve as donors for the edited target.

- **Fire is naturally partial.** It flickers in and out, so `frame_replace_precompute.py` can simply
  generate from a fire prompt, run the fire detector per frame, and keep the clips that happen to
  contain both fire frames and fire-free frames.
- **Nudity is not.** A naked body is present for the whole clip. There are no donor frames, so no
  target can be built. This is exactly why frame_replace did not transfer to nudity, and why the
  early single-prompt attempts (`prompts/partial_nudity*.csv`, one prompt asking for a
  clothed→naked transition) were unreliable — the model mostly ignores the transition.

Most concepts behave like nudity, not like fire. So instead of hoping for partiality, we
**manufacture** it.

## 2. The split-prompt sampler

Each row of the prompt CSV is an **A/B/C triple** describing the same scene, differing only in the
concept:

- **A** — the concept prompt (what we actually want to erase),
- **B** — the concept-free / "safe" counterpart,
- **C** — a neutral prompt shared by both halves.

All three (and the combined clip) share one initial noise per row, so they are directly comparable
and A/B double as a paired same-seed donor baseline.

Sampling runs in two phases:

1. **Split phase** (the first `split_step_frac` of the schedule — the content-setting steps).
   Each step does **two transformer forwards**, one conditioned on A and one on B, and splices the
   predictions per latent-frame region: frames on one side of `split_latent_frame` follow the
   concept prompt, frames on the other side follow the safe prompt. This is MultiDiffusion-style
   region conditioning, done in the *temporal* axis. Crucially only **one scheduler step** is taken,
   on the full spliced latent — running two schedulers would desynchronize the DPM-solver's internal
   multistep state and corrupt the trajectory.
2. **Heal phase** (the remaining steps). All latent frames are conditioned on the shared neutral
   prompt C. Since the late steps refine detail rather than set content, the concept/safe layout
   survives while the temporal seam between the two regions gets smoothed into one coherent clip
   (same person, pose, scene, lighting).

No attention surgery is involved — this is pure conditioning control, so it works on any T2V model
with classifier-free guidance.

**Key knobs**: `split_latent_frame` (where the boundary sits) and `split_step_frac` (how long the
split phase lasts). Both are decisive:

- split phase too short / C phase too long → both halves collapse to one state, the concept washes
  out entirely;
- split phase too long / C phase too short → a visible hard seam, two clips glued together;
- concept can also leak into the safe half if guidance is high.

The default is **0.85**, picked by the exp074/exp076 sweep.

**How much `split_step_frac` matters depends on what prompt C removes.** Re-scoring exp074's and
exp076's nudity sweep for seam contrast (see below) shows it is nearly flat above 0.3:

| `split_step_frac` | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| two-state clips | 2/5 | 4/5 | 4/5 | 4/5 | 4/5 | 4/5 | 4/5 | 4/5 | 4/5 | 4/5 | 4/5 |
| median seam ratio | 6.4 | 13.7 | 14.4 | 15.9 | 14.7 | 13.9 | 13.9 | 14.1 | 14.2 | 14.1 | 15.1 |

Only 0.2 degrades. So for nudity, a long heal phase is nearly harmless — and the reason is visible in
the prompts: nudity's C ("a person standing upright in a studio") **keeps the subject**, and only the
clothed/naked attribute is left unspecified. The heal phase has little to erase.

For the object classes it is the opposite. Church's C is "a green English village on a clear
afternoon" and chain saw's is "a wooden workbench in a cluttered garage" — **C removes the object
entirely**, so every heal step actively argues against the concept surviving in the concept half.

**But raising `split_step_frac` does not fix that, because above ~0.5 the knob has no authority over
content.** exp099 ran the same five seeds at 0.5 and at 0.85 and the clips came out near-identical:
2–4 grey levels apart over the whole clip, every two-state/collapsed verdict unchanged, `p3_s3202` at
median frame-difference 11.592 against 11.596. Seventeen steps of completely different conditioning
changed texture and not subject. The reason is ordinary diffusion behaviour — content is committed in
roughly the first 20 of 50 steps, and a conditioning switch placed after that only refines what is
already decided.

That single fact reconciles the two tables above. The nudity sweep is flat from 0.3 to 1.0 because
almost that whole range sits *after* the decisive window; only 0.2 degrades, because only 0.2 puts
the switch inside it. exp074's "0.2/0.3 wash the concept out" and exp099's "0.5 and 0.85 are the same
clip" are one observation seen from two sides.

**Practical rules:**

- **Do not expect `split_step_frac` above ~0.5 to change anything.** exp066/exp067's rebuild raised it
  0.5 → 0.85 on the strength of the C-deletion argument above and yield did not move (7/30 and 3/30).
  Anything in [0.5, 1.0] is the same experiment.
- **The decisive window is below ~0.4** — but a content-neutral tail does not buy it back.
  `tail_prompt_mode: "empty"` conditions the heal phase on the empty string, so under CFG the
  positive and negative embeddings coincide, the guidance term vanishes, and the tail becomes pure
  unconditional denoising: it heals the seam without arguing for or against any content. The
  hypothesis was that this would make a long tail safe for object classes, whose C deletes the
  object. **exp119 tested it on a 2x2 and rejected it.**

  | `split_step_frac` | `tail_prompt_mode: c` | `tail_prompt_mode: empty` |
  |---|---|---|
  | 0.3 | 3/5 | **2/5** |
  | 0.85 | 4/5 | 4/5 |

  The 0.85 arms are identical row-for-row, with contrast indices agreeing to within 0.003 — an
  independent confirmation of the inertness finding above, on a different axis. And at 0.3 `empty`
  was *worse*, not better: `p0_s3202` lost the concept entirely under `empty` (peak 0.044) while `c`
  kept it (0.247). So prompt C's concept-deleting content is not why an early split washes the
  concept out. What matters is how many decisive steps the concept region gets conditioned on prompt
  A, and at 0.3 it does not get enough — the tail's *content* is beside the point.

  **Keep `split_step_frac: 0.85` and `tail_prompt_mode: c`, and stop sweeping this axis.** Three of
  the sampler's knobs are now measured dead for content.
- **The levers that actually move object yield are the prompts**, not the sampler. See §3.2. The one
  sampler knob not yet ruled out is `concept_guidance_scale` (§3.3) — and it is aimed at a different
  mechanism than any of these.

### Measuring the collapse: seam contrast, not motion

The first failure mode is the one that silently produces useless training targets, so it is worth
measuring rather than eyeballing. The instinct is to check whether the clip moves — and that is
wrong. **A split clip that is nearly static within each half is a perfectly good target**, because
the supervision lives in the *difference between the halves*, not in motion. What kills a target is
collapsing to a single state across all 49 frames: the concept half and the safe half show the same
thing, so `x0_edited` ≈ `x0_original` and there is nothing to learn.

Averaging frame-to-frame differences conflates the two. exp066's `p25_s3226` has a median
frame-to-frame difference of 0.470 — indistinguishable from a frozen clip by that measure — and a
17.03 step exactly at its seam; it is the best clip in its batch.

`tools/check_seam_contrast.py` measures the right thing, in pixel space, from the `videos/*.mp4` that
`pull_results.sh` already downloads (no GPU, no VAE, no `.pt` off the cluster). Per clip it takes the
largest consecutive-frame difference, where it falls relative to the construction seam
(pixel frame `1 + 4*(sf-1)`), and how far it stands above the within-half median:

- **two-state** — one dominant transition, at the seam. What we want.
- **collapsed** — no transition anywhere; one state; no erase signal.
- **diffuse** — motion spread across the clip with no seam standing out. Either the halves never
  separated, or there is so much motion that the boundary is smeared away (exp067's `p16_s3317`:
  median 17.6, max/median 1.2).

This complements `tools/check_latent_motion.py`, which answers a different question in latent space —
is the *donor fill* frozen — and needs the latents.

**Known limitation: it is a whole-frame measure, and it is concept-blind.** Both limits were measured
on the object rebuild, and together they mean seam contrast must not be used to *select* training
rows:

- The step at the seam is a mean over all pixels, so a concept occupying a small share of the frame
  produces a smaller step even when the split worked perfectly. Nudity swaps most of the subject and
  scores a median seam ratio of ~14; church swaps one building inside an otherwise identical village
  and scored 3.8. exp067's `p27_s3328` — the one church clip that split *correctly*, bell tower
  present for 24 frames and gone thereafter — is scored **diffuse** (ratio 3.0), a false rejection.
- Two distinct states are not two states *of the concept*. exp066's `p13_s3214` scores a textbook
  two-state seam: a flower pot in the first half, a chain-and-hook object in the second. Peak
  p(chain saw) over all 49 frames is 0.003. A false acceptance.

So: **use `tools/screen_split_dataset.py` (§3.1) to decide what to train on, and seam contrast to
diagnose why a row failed.** Read the two-state fraction as a within-concept comparison, never as an
absolute bar across concepts.

## 3. From split clip to frame_replace dataset

`frame_replace_split_precompute.py` chains the sampler into a dataset, mirroring the fire builder:

```
A/B/C triple + seed
  → generate_split_clip()            # combined partial-concept clip
  → concept latent mask              # known by construction: [sf:] or [:sf], see below
  → edit_mask = concept mask + boundary_margin  # push the donor further from the boundary
  → edit_latent_reflected(..., edit_mask)  # concept block ← reflected/bouncing fill, not a freeze
  → save x0_original + x0_edited (+ optional MP4s)
  → per-frame detection (logging only, does not gate keep/skip)
```

**The concept mask is derived directly from `(split_latent_frame, concept_region)`, not from
detection** (fixed 2026-08-04 — see exp078's notes.md). Unlike the fire builder, where fire's
position is unpredictable and must be found by a detector, split-prompt chooses the split point
itself, so the mask is known before generation even starts: frames `[sf:]` (`concept_region:
"second"`) or `[:sf]` (`"first"`). An earlier version rederived the mask from NudeNet per-frame
confidences instead, which made yield hostage to the detector's known unreliability — flickering
mid-clip on static scenes, near-total misses on close-up crops, and over-triggering on multi-person
scenes (one person's clothed frames still scoring "concept present," killing the donor half
entirely). The detector still runs and its `frame_confidences` are logged in `metadata.json` for
human review, but it no longer decides what gets kept.

**`boundary_margin` (added 2026-08-04):** because split-prompt's concept block always touches a
clip edge (never flanked by safe frames on both sides), `edit_latent`'s two-sided interpolation
never actually engages here — it always hits the one-sided fallback, copying the *single* safe
frame nearest the boundary across the *entire* concept block. That makes the cleanliness of that
one frame disproportionately important: the heal phase (after `split_step_frac`) jointly attends
over the whole latent conditioned on prompt C, so a frame right next to the boundary can carry some
bleed from the other side even though the split phase's conditioning was cleanly separated.
`boundary_margin` (default 2) excludes that many latent frames closest to the boundary from being
used as the donor, so the copy comes from further inside the safe region instead. The true
construction mask is still logged as `concept_latent_mask`; what was actually replaced (mask +
margin) is `edited_latent_mask`.

**`edit_latent_reflected` (added 2026-08-04, replaces `frame_replace_ops.edit_latent` for this
script):** `boundary_margin` alone only changes *which* frame gets frozen — the frame-replace
`edit_latent`'s docstring itself warns that a hard single-frame copy taught the model to hold still
and globally suppressed motion (exp055: concept -84%, unrelated -29%), and split-prompt's
construction hits that exact fallback on *every* clip, not as an edge case. Instead of freezing one
donor across the whole block, `edit_latent_reflected` mirrors the safe segment's motion into the
concept region — position 0 (nearest the boundary) maps to the safe frame immediately adjacent to
it, then the source index walks deeper into the safe region as fill position moves away from the
boundary, bouncing back and forth (reflect/boomerang, like `scipy`'s `reflect` padding mode) if the
block is longer than the safe segment. The seam itself is unchanged (still a near-identical-content
cut at the boundary); only what fills in *away* from the seam changes, from frozen to mirrored
motion. `frame_replace_ops.edit_latent` itself is untouched — fire's naturally-flanked-on-both-sides
concept blocks (`frame_replace_precompute.py`, `unlearn_frame_replace_online.py`) still use its
two-sided interpolation, which is the right tool there.

The **training prompt stored with each target is the plain concept prompt A**, never the split
construction — at inference time that is the prompt we want to be safe.

Targets are dropped (recorded in `skipped.json`) only when:

- `insufficient_donor_frames` — the known concept-free side, after `boundary_margin` is excluded,
  has fewer than `min_donor_frames` latent frames — checked before generation, so it costs no GPU
  time.

A row can still be a *bad* training target without being skipped — e.g. a scene that renders badly
regardless of prompt (see exp074's seed-3163 finding, a persistent per-seed generation defect). Since
the mask stopped being detector-derived, **nothing inside precompute filters for quality at all**, so
selection is a separate step after the build: §3.1.

### 3.1 Selecting rows: the within-clip differential

`tools/screen_split_dataset.py` decides which built rows are worth training on. It is concept-
agnostic and reads only what precompute already logged, so it needs no GPU and no second job.

The obvious screen — "did the detector fire?" — is what cost exp066/exp067 their first run. An
absolute threshold asks `p(church) > 0.03?` on a scale that is not comparable across scenes: over
exp067's 30 clips the per-clip peak p(church) spans 0.0009 to 0.357, driven mostly by framing and
lighting. Any single cut through that range is arbitrary.

The question that *is* well posed is paired and within-clip: **does the half conditioned on prompt A
read more concept than the half conditioned on prompt B?** Both halves share a seed, a scene, a camera
and a lighting setup, so everything except the concept cancels. The tool reports it as a bounded
contrast index

```
ci = (mean(concept_half) - mean(safe_half)) / (mean(concept_half) + mean(safe_half))     ∈ [-1, 1]
```

and requires three things of a row, because no one of them is sufficient (§2's two false verdicts,
plus the blank-target case below):

| gate | rejects | verdict when it fails |
|---|---|---|
| `--max-degenerate-frac` | the edited target never rendered | `blank-target` |
| `--min-concept-max` | prompt A never rendered the concept anywhere | `no-concept` |
| `--min-contrast-index` | it rendered, but the safe half has it too | `not-split` |

Defaults are 0.1, 0.10 and 0.4. The contrast threshold sits inside a clear gap in the church data (the
three genuine splits score 0.87 / 0.63 / 0.49, the next row down 0.18) and keeps every clip that
survived visual review in both classes. Thresholds are CLI arguments and not a per-concept table on
purpose: `build_detector` is the one place the codebase maps a concept string to behaviour, and this
must not become a second one.

**The blank-target gate, and why the differential needs it.** A blank frame scores p(concept) ≈ 0
exactly like a legitimately concept-free one, so a clip whose safe half never rendered gets a *perfect*
separation score — the blanker it is, the better it screens. It then passes into `edit_latent_reflected`,
which mirrors that blank half into the concept region, and the result is a training target that is
mostly white. Two rows were found this way on 2026-08-16, both church: exp122's `p22_s3353` (contrast
index **+0.994**, edited target **49/49 blank frames**) and exp118's `p4_s3305` (36/49 blank, and the
frames that are not blank still show a church) — the second of which had already trained in exp070.

So the gate is checked **first** and it reads the `_edited` clip, i.e. the thing training actually
regresses onto, where the other two gates read the source clip's logged confidences. That costs a
video decode, so it is skipped with a loud warning when the videos are not next to the metadata
(`--videos-dir` overrides, `--no-blank-check` disables). Reuses
`zml/benchmarks/frame_quality.py::degenerate_frame_mask`, the same structure test used for eval clips.

`--write-filtered` writes the surviving entries to the **experiment root**, not under `outputs_*/`,
which is gitignored — a filtered set living there never reaches the cluster (the mistake that aborted
exp085).

`tools/screen_split_face_dataset.py` is the absolute-threshold-only ancestor of this tool, kept
because exp115/exp116's published keep-lists were selected with it. New work should use the general
one.

### 3.2 Prompt framing decides yield, not the sampler

Screening exp066/exp067's rebuild gives a blunt result:

| | rows | pass | `not-split` | `no-concept` |
|---|---|---|---|---|
| exp066 chain saw | 30 | 7 (23%) | 6 | **17** |
| exp067 church | 30 | 3 (10%) | 10 | **17** |

`no-concept` means the base model never drew the object anywhere in the clip. **The same 17 of 30 in
both classes** — a shared, structural cause, not per-class bad luck. The splitter cannot separate a
concept that was never rendered, so most of what looked like a sampler problem was never one.

The face thread hit this first and measured the fix. exp115 kept 9/30; 14 of the 21 rejects had
`original_max_confidence` at or near 0, in wide, side-on or occluded framings. exp116 rewrote the
prompts with controlled medium/close frontal framing, held everything else fixed, and yield went
**30% → 50% and 63%** — while a re-seed of the original prompts reproduced 30% exactly, proving it was
framing and not seed luck.

The object prompts had the same two defects, both visible against the eval set the base model scores
0.739 (church) and 0.506 (chain saw) top-1 on:

1. **Framing.** "Static wide shot of a small church across a field of wildflowers" puts a small
   building in a large landscape. A detector classifying a 224px view of the whole frame is not being
   obtuse — the frame genuinely is not *of* the object.
2. **Specificity.** Eval prompts name the class-identifying parts ("with a tall steeple", "its orange
   casing and bar clearly visible"); the split prompts said only "a church" / "a chain saw".

`tools/build_split_imagenet_closeup_prompts.py` applies both, and applies the second **symmetrically
to prompt B** — if only A gains detail, B loses the splice on prompt strength rather than on content,
which buys yield by quietly turning the safe half into the concept half. exp117/exp118 tested it with
the settings, seeds and prompt C held verbatim, and it reproduced the face thread's result:

| | pass before | pass after | `not-split` | `no-concept` |
|---|---|---|---|---|
| chain saw (exp066 → **exp117**) | 7 (23%) | **14 (47%)** | 6 → 4 | 17 → 12 |
| church (exp067 → **exp118**) | 3 (10%) | **14 (47%)** | 10 → 5 | 17 → 11 |

Church's `not-split` half has its own confirmed cause and cure: exp067's substitute buildings were
church-shaped, and rewriting them to have no tower, spire or bell-cote took whole-clip prompt B from
a peak p(church) of 0.247 — tying the concept half — down to 0.064 across all 30 rows.

**Rule for a new concept: check that the base model renders it under the training prompts before
touching any sampler knob.** `emit_whole_clip_target: true` gives that for free — the A-side
confidences come from a plain generation, so they separate a prompt failure from a splitter failure
within the same job.

### 3.3 After the prompts: the splice suppresses concepts the prompt does render

exp117's whole-clip diagnostics moved the object thread's diagnosis, and the finding is general
enough to expect on the next concept. With the reframed prompts, plain prompt A renders the object in
**29 of 30 chain-saw rows and 28 of 30 church rows** — "the base model never drew it" is no longer the
bottleneck. The remaining loss is the *splice* killing a concept that the identical (prompt, seed)
renders fine unsplit, and it fails binary. Split concept-half mean over the same row's plain-A mean:

| rows | chain saw | church |
|---|---|---|
| passing | 1.12 | 0.76 |
| failing | **0.06** | **0.23** |

The mechanism is in `generate_split_clip`: `pred_a` and `pred_b` are each predicted over the *whole*
latent and only the prediction is spliced, so `pred_a` is evaluated in a context whose other region
is converging on prompt B. CogVideoX's temporal-coherence prior then argues the clip is one scene and
pulls the concept region toward the substitute. Either the object establishes itself early enough to
hold its half or it is gone — hence no middle.

**exp120 tested this and produced a split verdict: the mechanism is confirmed, the obvious knob is
not the cure.** `concept_guidance_scale` raises CFG on the concept branch alone, at zero extra
compute. Swept over the 12 exp117 rows that failed *despite* plain A rendering the object, with the
base value as a control arm:

| concept guidance | renders the concept | passes the screen |
|---|---|---|
| 6.0 (control) | 0/12 | 0/12 |
| 9.0 | **7/12** | 2/12 |
| 12.0 | 4/12 | 3/12 |

The control reproduced 0/12 exactly. At 9.0 seven rows render the object again — so the pull is real
and conditioning strength does fight it — but five of those seven then read as concept in **both**
halves (`p0_s3203`: concept half 0.471, safe half 0.470). The concept comes back and immediately
bleeds across the seam, so yield barely moves. Response is also non-monotone per row (`p2_s3206`:
0.633 at 9.0, 0.0004 at 12.0), meaning part of what the scale changes is which sample you draw.
Keep `concept_guidance_scale: None`.

### 3.3.1 `split_mode`: splice the trajectories instead

If both halves of exp120's result come from the two regions sharing one latent, then the fix belongs
in the context rather than in the guidance scale. `Config.split_mode` (added 2026-08-16) offers:

- **`prediction`** (default; every dataset up to exp122) — one latent, two predictions per step, the
  *prediction* spliced. Under CogVideoX's element-wise scheduler step this is arithmetically the same
  as splicing the latent, so the only thing it changes is what the transformer sees.
- **`trajectory`** — two latents from the same initial noise, each denoised under its own prompt for
  the whole split phase, spliced **once** at `split_step`, then healed by the tail as before. `pred_a`
  is never evaluated in a B-converging context and the safe region never sees prompt A.

Cost is identical: two transformer calls per split step either way. The thing to watch is coherence
across the seam, which currently comes from the shared noise *and* the shared latent — §2's exp076
finding (the cut is hard at every `split_step_frac`, including with zero heal steps) is the reason to
expect the noise carries most of it, and exp127 is the test, with six currently-passing rows in its
CSV as the regression check.

### 3.4 The whole-clip variant is a diagnostic, not a training target

`emit_whole_clip_target` pairs prompt A's plain clip with prompt B's plain clip at the same seed. It
is tempting as a seam-free target on the argument that the two differ by one noun under one seed.
Measured on exp117, that argument fails: mean per-pixel |A − B| at the same seed is **56.5**, against
**72.4** between clips of two *unrelated* rows, with 74% of pixels moving more than 25 levels (church:
52.9 vs 86.4). The noun swap redraws the frame — a shared seed does not hold the scene. Training on
it would teach a global scene substitution, the opposite of frame_replace's minimal-edit premise.

Turn it on for a new concept's *first* build, where it separates a prompt failure from a splitter
failure in one job (§3.2), and off for every build after — it costs 2.1x runtime.

## 4. De-biasing: avoiding the positional shortcut

A naive split dataset always puts the concept in, say, the second half. The trainer can then satisfy
the loss by learning "copy the first half onto the second half" — a *positional* rule that has
nothing to do with the concept and will not generalize to full-concept prompts.

Two knobs break that correlation:

- **`concept_region`** — `first` / `second` / `random`. With `random`, roughly half the dataset has
  the concept early and half late.
- **`split_jitter`** — moves the boundary by ±N latent frames, seeded per clip, so the edit is not
  anchored to a fixed index.

Together these make concept *position* uninformative, so the only consistent rule that explains the
targets is "remove the concept".

**The shortcut check is an eval-time check, not a training-loss check.** Training loss will look fine
either way. What discriminates is evaluating on *fully*-concept prompts (e.g. plain full-nudity
prompts): there is no concept-free half to copy, so a model that only learned the positional rule
will not erase, while a model that learned the semantics will.

Watch also for the exp055 failure mode: if a concept block is *terminal* (touches the clip
boundary), the interpolated donor can degenerate into a frozen frame, killing motion.

## 5. Status (nudity)

| Experiment | What it did | Outcome |
|---|---|---|
| exp059 / exp060 | Generate A/B/C + combined clips, inspect the splice | Splice works — coherent clip, clothed early / naked late |
| exp061 (run 1) | First full nudity frame_replace dataset, 30 triples (`prompts/split_nudity.csv`, seeds 3101–3130) | 20/29 auto-kept (row 29/seed 3130 never processed — precompute likely cut short); manual review then dropped 8 more bad splices/edits → **12/29 confirmed-good** |
| exp062 (run 1) | Pilot training on the 20-auto-kept dataset (exp057 eta=2 regime, `concept: nudity`) | `nudity_detection_rate` 0.1→0.6(step500)→0.4(step600); unrelated held at 0 detection / clip ~0.33 |
| exp062 (run 2) | Retrain on the 12 human-confirmed-good triples | Running |
| exp061 (run 2) | Dataset felt too small at 12 — extended `split_nudity.csv` to 52 triples (kept the 12, added 40 new seeds 3131–3170, same template/knobs) | 31/52 auto-kept (12 originals reproduced deterministically + 19/40 new); row 51/seed 3170 missing entirely again (2nd time — likely a real "last row" bug); human review of the 19 new approved 9 → **21/52 confirmed-good total** |
| exp062 (run 3) | Retrain on the 21 confirmed-good triples (`outputs_20260802_223148`) | Pending |

If exp062 erases on full-nudity prompts with collateral held, the next step is to scale
`split_nudity.csv` further. If it erases only the partial training clips, revisit de-biasing (more
`concept_region` mixing, concept-in-the-middle layouts). Also worth investigating: the auto-kept
yield has been low and human review knocks it down further (run 2's new triples: 47.5% auto-kept,
then only 47% of those passed review, ~22.5% overall) — tuning `split_step_frac`/`split_latent_frame`
could reduce wasted generation before scaling much further. The disappearing-last-row bug (rows
29 and 51, both times the final CSV row) is **root-caused and fixed** (2026-08-04): in
`frame_replace_split_precompute.py`, a skipped row hit `continue` before reaching the
`metadata.json`/`skipped.json` writes, which only ran on the kept path — so a run whose trailing
rows were all skips never flushed them to disk (silently correct in memory, silently wrong on
disk). Rebuilding a dataset whose current `skipped.json` predates this fix may undercount skips at
the tail; treat pre-fix skip counts near the end of a CSV as suspect.

## 6. Generalizing to a new concept

split-prompt is concept-agnostic. The cost of a new concept is exactly two things:

1. an **A/B/C prompt CSV** (with per-row seeds — see the seed policy in `CLAUDE.md`), and
2. a **per-frame detector** for the concept in `zml/benchmarks/` — used for eval/reporting and
   logged during dataset construction, but (as of 2026-08-04) no longer required to be accurate
   enough to gate dataset keep/skip, since the mask itself is now built from the known split point.

Everything else — sampler, mask construction, `edit_latent`, the trainer — is unchanged.
See [`comparison_targets.md`](comparison_targets.md) for which concepts are worth attacking next.

**Write the CSV against the eval prompts, not from scratch.** Both threads that have transferred to a
new concept lost most of their first dataset the same way — to prompts under which the base model
does not render the concept at all (faces: 14 of 21 rejects in exp115; objects: 17 of 30 rows in each
of exp066/exp067). The prompt A of a training row and the eval prompts should be alike in framing and
in how specifically they name the concept; where they are not, the training set is measuring
something the eval never asks about. §3.2 has the two concrete defects and the fix.

**Budget one cheap screening pass, always.** Set `emit_whole_clip_target: true` on the first build of
a new concept and run `tools/screen_split_dataset.py` (§3.1) before anything trains. The A-side
confidences distinguish "the prompts are wrong" from "the splitter is wrong", which are the two
failures a new concept actually hits, and they cost one job instead of two.

**One open question about the CSVs themselves.** Every split CSV we have written so far ends all three
of A, B and C with the same scaffold: `"Static shot … The camera is fixed and never moves."` — 30/30
rows in both `split_imagenet_*.csv` and 52/52 in `split_nudity.csv`. It was a reasonable choice (a
fixed camera keeps the subject in a stable screen position, so the temporal splice reads cleanly), and
since a static two-state clip is a perfectly good target it is not a defect. But it does bound what the
method has been shown to do, it makes the training prompt A stylistically unlike the eval prompts,
and `edit_latent_reflected` has since changed the trade-off — it *mirrors* the safe segment's motion
into the concept block, so motion in the safe half is now an asset rather than a liability. Whether
split-prompt can stitch two prompts that both carry motion, or whether motion smears the seam away
(exp067's `p16_s3317`), was exp099.

**exp099 answered it: keep the static scaffold.** Motion-carrying prompts scored **0/5** two-state
against the static arm's 2/5, at both `split_step_frac` values, and their median seam ratio collapsed
to 1.1 — the `p16_s3317` failure generalised. Two of the five motion clips also came out *more* static
than their static-prompt counterparts (median frame difference 0.045 and 0.054), so asking for camera
motion is not even a reliable way to get it. The scaffold stays in every split CSV.
