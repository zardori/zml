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
  → decode + per-frame detection     # zml/benchmarks/ (NudeNet for nudity)
  → concept latent mask              # per latent-frame boolean
  → edit_latent(..., interpolate)    # concept frames ← interpolated donor frames
  → save x0_original + x0_edited (+ optional MP4s)
```

The **training prompt stored with each target is the plain concept prompt A**, never the split
construction — at inference time that is the prompt we want to be safe.

Targets are dropped (recorded in `skipped.json`) when:

- `no_concept` — the splice did not actually render the concept (detector found nothing);
- `insufficient_donor_frames` — fewer than `min_donor_frames` concept-free latent frames remain.

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
| exp061 | First full nudity frame_replace dataset, 30 triples (`prompts/split_nudity.csv`, seeds 3101–3130) | **20/29 kept**; skips split between `no_concept` and `insufficient_donor_frames`; balanced 9 `first` / 11 `second` |
| exp062 | Pilot training on that dataset (exp057 eta=2 regime, `concept: nudity`) | Decides whether erasure transfers + whether the shortcut is gone |

If exp062 erases on full-nudity prompts with collateral held, the next step is to scale
`split_nudity.csv` and rebuild a larger dataset. If it erases only the partial training clips,
revisit de-biasing (more `concept_region` mixing, concept-in-the-middle layouts).

## 6. Generalizing to a new concept

split-prompt is concept-agnostic. The cost of a new concept is exactly two things:

1. an **A/B/C prompt CSV** (with per-row seeds — see the seed policy in `CLAUDE.md`), and
2. a **per-frame detector** for the concept in `zml/benchmarks/`.

Everything else — sampler, mask construction, `edit_latent`, the trainer — is unchanged.
See [`comparison_targets.md`](comparison_targets.md) for which concepts are worth attacking next.
