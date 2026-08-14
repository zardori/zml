---
status: ready
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  2x2 over tail_prompt_mode [c, empty] x split_step_frac [0.3, 0.85] on 5 seeds. For object classes
  prompt C is necessarily a concept-*deleting* prompt, so every heal step argues against the concept
  the A-half should keep; "empty" makes the tail pure unconditional denoising instead. exp099 showed
  the tail is inert for content above ~0.5, so the honest prediction is that the 0.85 arms are
  identical and only 0.3 can move. Not submitted yet.
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

## Status
- [ ] Submitted.
- [ ] Scored; the four cells compared against the predictions above.
- [ ] `docs/split_prompt.md` updated with the outcome either way.
