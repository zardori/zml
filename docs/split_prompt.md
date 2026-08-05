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
regardless of prompt (see exp074's seed-3163 finding, a persistent per-seed generation defect). That
kind of failure isn't reliably catchable by the detector either, so it's caught by human review of
the kept set before training, same as before.

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
