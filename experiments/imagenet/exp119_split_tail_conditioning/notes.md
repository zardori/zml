---
status: done
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  Hypothesis rejected, cleanly. The 0.85 arms are identical (4/5 both, same rows, contrast indices
  within 0.003), confirming exp099's inertness finding. And `empty` did NOT rescue an early split:
  0.3-empty scored 2/5 against 0.3-c's 3/5, so prompt C's concept-deleting content is not why a low
  split_step_frac washes the concept out. The tail is not a lever at any setting — keep
  `split_step_frac: 0.85` + `tail_prompt_mode: c` and stop sweeping this axis.
---
# exp119 — what should condition the heal phase?

## The question
After `split_step_frac`, both halves switch to the shared neutral prompt C. For an object class C is
necessarily the *object-removed* scene — chain saw's is "a wooden workbench in a cluttered garage",
church's is "a green English village". So the heal phase is not neutral at all: every step of it
pushes against the very concept the A-half is supposed to be keeping.

Nudity never exposed this. Its C keeps the subject and leaves only the clothed/naked attribute open,
so there is little for the tail to delete — which is why sweeping `split_step_frac` there (exp074,
exp076) looked flat and reassuring, and why that reassurance did not transfer.

`tail_prompt_mode: "empty"` conditions the tail on the empty string. Under CFG the positive and
negative embeddings then coincide, the guidance term vanishes, and the tail becomes **pure
unconditional denoising**: it sharpens whatever the split phase committed to without arguing for or
against any content.

## What we already know, and why `split_step_frac` is the second axis
exp099 ran these same five seeds at 0.5 and 0.85 and the clips came out near-identical — 2-4 grey
levels apart, every two-state/collapsed verdict unchanged, `p3_s3202` at median frame-diff 11.592 vs
11.596. Seventeen steps of different conditioning changed texture and not subject. Content is
committed in roughly the first 20 of 50 steps, so a switch placed after that only refines what is
already decided.

exp074's finding that 0.2/0.3 wash the concept out is the same fact from the other side: those put
the switch *inside* the decisive window, where the tail prompt does win.

So the tail-mode axis can only show anything where the tail has authority, which is why 0.3 is here
and 0.5 is not.

## Predictions, written down first
- **0.85-c ≈ 0.85-empty.** Both arms inert; anything else contradicts exp099.
- **0.3-c washes the concept out**, reproducing exp074.
- **0.3-empty is the interesting cell.** If prompt C's content is why a long tail erases, an empty
  tail at 0.3 should keep the concept *and* get 35 steps of seam healing — plenty of steps to merge
  the halves, none of them deleting. That would make a long empty tail the new default for objects
  and would be the first knob on this sampler that actually moves object yield.

## Reading it
| outcome | means |
|---|---|
| 0.3-empty ≫ 0.3-c | hypothesis confirmed; long empty tail becomes the object default |
| all four alike | the tail is not a lever at all; the prompts are (exp117/exp118) |
| 0.85-c ≠ 0.85-empty | contradicts exp099's inertness finding — re-examine that before believing anything else here |

## Setup
5 rows from `prompts/imagenet_objects/split/chain_saw_closeup_sweep.csv`, a verbatim subset of exp117's CSV
so a result here transfers to that build directly. `concept_region: second` and `split_jitter: 0`
fixed so the seam sits at the same index in every clip.

**Why these five seeds.** 3202/3204/3208/3210/3217 are the exp066 run-2 rows with the highest peak
p(chain saw) — 0.61 to 0.81. A sampler sweep should ask "does this knob preserve a concept the model
already renders"; on rows chosen for rendering that question has a clean answer, whereas on rows
chosen at random it is confounded with the 17-in-30 chance the object was never drawn at all. (This
is a different selection from exp099's five, which were picked to span *splice* outcomes.)

**Why chain saw only.** The arms are scored with `tools/screen_split_dataset.py`, which reads the
per-frame detector confidences, and a detector build takes a single `concept_target`. Scoring on the
detector rather than on seam contrast is deliberate: church's one correct split (exp067 `p27_s3328`)
is invisible to a whole-frame pixel measure, so pixel-space scoring would have mismeasured this.

Run as `method: frame_replace_split` rather than `split_prompt` for the same reason — only that method
runs the detector and logs confidences. The edit and the saved latents are incidental at 5 rows.

`./submit_job.py helios experiments/imagenet/exp119_split_tail_conditioning/config.yaml` — 4 jobs x 5 rows,
~15 min each.

## Results (`grid_20260815_014932`, helios, 4 jobs x 5 rows)

| run | `split_step_frac` | `tail_prompt_mode` | pass | no-concept | not-split |
|---|---|---|---|---|---|
| 001 | 0.3 | c | 3/5 | 2 | 0 |
| 002 | 0.3 | empty | **2/5** | 3 | 0 |
| 003 | 0.85 | c | 4/5 | 0 | 1 |
| 004 | 0.85 | empty | 4/5 | 0 | 1 |

Read against the predictions written above:

- **0.85-c ≈ 0.85-empty: confirmed, and more tightly than expected.** Same four passing seeds, same
  failing seed, contrast indices agreeing to within 0.003 (`p0_s3202` +0.970 vs +0.968, `p3_s3210`
  +0.633 vs +0.612). Seventeen steps of completely different tail conditioning changed nothing about
  content. exp099's inertness finding stands.
- **0.3 is worse than 0.85 under both tail modes**, reproducing exp074 from the other side.
- **0.3-empty ≫ 0.3-c: rejected.** It went the other way — 2/5 against 3/5, with `p0_s3202` losing
  the concept entirely under `empty` (peak 0.0436) while `c` kept it (0.2467). So prompt C's
  concept-deleting content is *not* why an early split washes the concept out. A tail that argues for
  nothing is no kinder to the concept than one that argues against it; what matters is that the
  concept region gets enough decisive steps of prompt A, and at 0.3 it does not.

## What this settles
The tail is not a lever at any setting we can reach. **Keep `split_step_frac: 0.85` and
`tail_prompt_mode: c`** — the defaults every current object config already uses — and stop spending
runs on this axis. Combined with exp099, three of the split sampler's knobs are now measured dead for
content, which is what redirects the thread to prompt framing (exp117/exp118, which worked) and to
the concept branch's own conditioning strength (exp120).

Caveat on power: 5 rows per cell, chosen for high base-model confidence. That is enough to reject a
"≫" hypothesis and enough to establish the 0.85 arms are identical (they agree row-for-row), but it
would not detect a small effect. Nobody should re-run it looking for one — the 0.85 agreement is the
result, and it is not a power question.

## Status
- [x] Submitted.
- [x] Scored; all four cells compared against the predictions above.
- [x] `docs/split_prompt.md` §2 updated with the outcome.
