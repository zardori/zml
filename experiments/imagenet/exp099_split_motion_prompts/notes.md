---
status: done
concept: imagenet
method: split_prompt/precompute
thread: imagenet
takeaway: >
  DONE, and it settled two questions. (1) Keep the static scaffold: motion-carrying prompts scored
  0/5 two-state against static's 2/5, median seam ratio 1.1 vs 7.0. (2) split_step_frac is INERT
  above ~0.5 — the same seed at 0.5 and 0.85 gives near-identical clips (2-4 grey levels apart,
  every verdict unchanged), because content is committed in the first ~20 of 50 steps. That kills
  the C-drops-the-object hypothesis as a reason to raise the knob and redirects the object thread
  to prompt framing (exp117/exp118) and to what conditions the tail (exp119).
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

- **Prompt style** — `prompts/imagenet_objects/split/static_sweep.csv` (rows copied verbatim from the
  exp066/exp067 masters) vs `prompts/imagenet_objects/split/motion_sweep.csv` (same five scenes and seeds,
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

`./submit_job.py helios experiments/imagenet/exp099_split_motion_prompts/config.yaml`

## Evaluation
`uv run python tools/check_seam_contrast.py --metadata <run>/outputs/metadata.json` per arm, locally,
CPU-only — it reads the sampler's metadata shape and the `_combined.mp4`s directly. Compare the
two-state fraction and median seam ratio across the four cells. Then watch the clips: the metric
cannot tell "the halves separated cleanly" from "the halves separated into something ugly", and per
exp074/exp076 the automated read has been wrong about this class of question before.

## Result (`grid_20260808_235454`, helios, ~16 min per arm)

`tools/check_seam_contrast.py` per arm:

| arm | two-state | diffuse | collapsed | median seam ratio |
|---|---|---|---|---|
| static, 0.5 | **2/5** | 2 | 1 | 5.2 |
| static, 0.85 | **2/5** | 2 | 1 | 7.0 |
| motion, 0.5 | 0/5 | 3 | 2 | 1.1 |
| motion, 0.85 | 0/5 | 3 | 2 | 1.1 |

### 1. Keep the static scaffold
The motion arm lost every two-state clip, and lost them in the way the risk section predicted: the
three that moved landed in `p16_s3317`'s diffuse regime (max/median 1.1 — no transition stands out
anywhere). `p2_s3328` and `p4_s3226`, the two clips that split cleanly under the scaffold with seam
ratios of 389 and 54, drop to 1.1 with motion prompts.

The other two are the more interesting failure: `p0_s3311` and `p1_s3322` came out **more static than
the static arm** (median frame difference 0.045 and 0.054, against 0.617 and 0.164). Asking for camera
motion is not a reliable way to get camera motion, so the scaffold is not even trading motion for seam
quality — it is trading nothing for seam quality.

Decision recorded in `docs/split_prompt.md` §6. The scaffold stays in every split CSV, including
`split_nudity.csv`.

### 2. `split_step_frac` is inert above ~0.5 — the bigger finding
The two `split_step_frac` cells were included as a control. They came back **near-identical**, which
was not the expected outcome and matters more than the motion axis:

| clip | mean abs. difference between the 0.5 and 0.85 clip | median frame-to-frame motion, for scale |
|---|---|---|
| `p0_s3311` | 2.725 | 0.662 |
| `p1_s3322` | 0.948 | 0.192 |
| `p2_s3328` | 2.419 | 0.230 |
| `p3_s3202` | 4.493 | 11.645 |
| `p4_s3226` | 2.865 | 0.366 |

Seventeen steps of completely different conditioning move the clip by 2–4 grey levels and change no
verdict. `p3_s3202`'s median frame difference is 11.592 at 0.5 and 11.596 at 0.85; `p4_s3226`'s seam
ratio is 54.0 and 56.1. The subject is identical; only texture moves.

The reading is ordinary diffusion behaviour: content is committed in roughly the first 20 of 50 steps,
so a conditioning switch placed after that only refines what is already decided. This reconciles the
two things that looked contradictory before — exp074's "0.2/0.3 wash the concept out" and exp076's
"0.85 through 1.0 are within noise" are one fact seen from two sides, and the boundary sits somewhere
below 0.5.

**What it invalidates.** The hypothesis that exp066's 17 `no_concept` rows came from prompt C deleting
the object over a long heal phase predicted 0.85 would beat 0.5 here. It did not, and the exp066/exp067
rebuild at 0.85 confirmed it at scale (yield did not move: 7/30 and 3/30 on screening). The C-deletion
mechanism is real but only has authority in the decisive window below ~0.4 — which is what exp119 now
tests with `tail_prompt_mode: "empty"`, and it means the object thread's real lever is prompt framing
(exp117/exp118), not this knob.

## Status
- [x] Submitted.
- [x] Seam contrast scored per arm.
- [x] Clips reviewed by eye.
- [x] Decision recorded in `docs/split_prompt.md` §6 (scaffold) and §2 (`split_step_frac` inertness).
