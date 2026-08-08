---
status: ready
concept: imagenet
method: split_prompt/precompute
thread: imagenet
takeaway: >
  Staged, not yet submitted. Asks whether split-prompt can stitch two prompts that both carry
  motion, or whether the static-camera scaffold every split CSV uses is load-bearing. 2x2 over
  {static, motion-carrying prompts} x split_step_frac {0.5, 0.85}, 5 paired scenes, scored with
  tools/check_seam_contrast.py.
---
# exp099 — can split-prompt stitch two moving prompts?

## Why
Every A/B/C CSV in the repo ends all three prompts with the same scaffold: `"Static shot … The camera
is fixed and never moves."` — 30/30 rows in `split_imagenet_church.csv` and
`split_imagenet_chain_saw.csv`, 52/52 in `split_nudity.csv`. It was a sensible default: a fixed camera
keeps the subject in a stable screen position, so the temporal splice reads cleanly.

It is **not** a defect, and this experiment is not a bug fix. A split clip that is static within each
half is a perfectly good training target, because the supervision lives in the difference *between*
the halves (see `docs/split_prompt.md` §2). But the scaffold bounds what the method has been shown to
do, and two things have changed since it was written:

- `edit_latent_reflected` (`603b4c3`) mirrors the safe segment's motion into the concept block instead
  of freezing one donor frame, so motion in the safe half is now an asset rather than a liability.
- The 20 eval prompts per class carry no camera instruction at all, so training prompt A is
  stylistically unlike what the LoRA is evaluated on.

## The specific risk
More motion is not obviously better. exp067's `p16_s3317` is the counterexample: median frame-to-frame
difference 17.6, max/median 1.2 — so much diffuse motion that the seam was smeared away entirely and
the clip held no distinguishable two-state structure. If motion-carrying prompts land in that regime,
the scaffold stays exactly as it is, and that is a perfectly good result.

## Setup
2x2 grid, both axes list-valued so `submit_job.py` produces four jobs of five rows.

- **Prompt style** — `prompts/split_imagenet_static_sweep.csv` (rows copied verbatim from the
  exp066/exp067 masters) vs `prompts/split_imagenet_motion_sweep.csv` (same five scenes and seeds,
  scaffold removed, an ambient or camera-motion cue added to all three of A, B and C). Verified paired:
  identical scenes, identical seeds, so prompt wording is the only difference between arms.
- **`split_step_frac`** — 0.5 (run 1's value) and 0.85 (the current default). Included so a
  prompt-style effect cannot be confused with the split_step_frac change the exp066/exp067 rebuild
  makes at the same time — **and because this axis answers a second open question on its own.**
  Re-scoring exp074/exp076 shows seam contrast is flat from 0.3 to 1.0 on *nudity*, whose prompt C
  keeps the subject. For objects, C drops the object entirely, so the heal phase should trade against
  concept survival in a way it never did for nudity. That is currently a hypothesis (it is the reading
  behind exp066's 17 `no_concept` rows) and these two cells test it directly, on five scenes, for
  under an hour of compute — much cheaper than inferring it from a 30-row rebuild.
- Five seeds chosen to span run 1's outcomes: **3328** and **3226** came out cleanly two-state,
  **3322** collapsed to a single state, **3311** is the clip whose detector flicker started this
  thread, **3202** was diffuse.

`split_jitter: 0` and `concept_region: second` are fixed so the seam sits at the same index in every
clip and the four arms are directly comparable — this is a measurement run, not a dataset build.

`./submit_job.py helios experiments/exp099_split_motion_prompts/config.yaml`

## Evaluation
`uv run python tools/check_seam_contrast.py --metadata <run>/outputs/metadata.json` per arm, locally,
CPU-only — it reads the sampler's metadata shape and the `_combined.mp4`s directly. Compare the
two-state fraction and median seam ratio across the four cells. Then watch the clips: the metric
cannot tell "the halves separated cleanly" from "the halves separated into something ugly", and per
exp074/exp076 the automated read has been wrong about this class of question before.

## What would change from the result
- Motion arm holds its two-state fraction → drop the scaffold from future concept CSVs, and revisit
  `split_nudity.csv`.
- Motion arm degrades toward `p16_s3317`'s diffuse regime → keep the scaffold, and record the reason
  in `docs/split_prompt.md` §6 so nobody re-opens it.
- 0.85 beats 0.5 on the object scenes → confirms the C-drops-the-object mechanism and settles the
  rebuild's main knob; flat, as on nudity → the 17 `no_concept` rows need another explanation, and
  the honest place to look next is the detector threshold against a small in-frame object.

## Status
- [ ] Submitted.
- [ ] Seam contrast scored per arm.
- [ ] Clips reviewed by eye.
- [ ] Decision recorded in `docs/split_prompt.md` §6.
