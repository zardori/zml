---
status: ready
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  exp120's prescribed follow-up: the concept-suppression pull is real but CFG is the wrong instrument,
  so remove the coupling instead of pushing against it. New split_mode: trajectory denoises prompt A
  and prompt B on separate latents from shared noise and splices ONCE at split_step, at identical
  cost. Tested on exp120's 12 suppressed rows (known 0/12) plus 6 exp117 survivors as the coherence
  regression. Not submitted yet.
---
# exp124 — splice the trajectories, not the predictions

## The question
`generate_split_clip` has always kept **one** latent: at each split-phase step it predicts under
prompt A and under prompt B over the whole tensor and splices the *prediction*. Under CogVideoX's
element-wise scheduler that is the same arithmetic as splicing the latent — the two differ in nothing
except what the transformer *sees*, and that is the whole point. `pred_a` is computed on a latent
whose other half is converging on prompt B, so the temporal-coherence prior argues the clip is one
scene and drags the concept region toward the substitute.

exp120 tested the obvious counter (more guidance on the A branch) and produced the cleanest possible
split verdict:

| guidance | renders the concept | passes the screen |
|---|---|---|
| 6.0 (control) | 0/12 | 0/12 |
| 9.0 | **7/12** | 2/12 |
| 12.0 | 4/12 | 3/12 |

The suppression is real and beatable — but the recovered concept then shows up in the *safe* half too
(5 of those 7 rows screen `not-split`), so pushing harder on conditioning trades one failure for
another. That is a coupling through the shared context, and the fix has to be in the context.

**`split_mode: trajectory`**: two latents from the same initial noise, each denoised under its own
prompt for the whole split phase, spliced once at `split_step`, then healed by the tail as before.
The concept region is denoised in a pure-A context, and the safe region never sees prompt A at all.
Cost is unchanged — two transformer calls per split step either way.

## The cost this might carry, and why it is worth testing anyway
Clip coherence across the seam (same scene, lighting and camera on both sides) currently comes from
two sources: the shared initial noise, and the shared latent during the split phase. This removes the
second. exp076 is the reason to expect the first is doing most of the work — it found the
clothed→nude cut is *hard* at every `split_step_frac` from 0.2 to 1.0, including with zero heal steps,
i.e. coherence was already fully present without any healing. If that reading is right, trajectory
mode costs little; if it is wrong, this experiment is where we find out, which is why 6 currently-
passing rows are in the CSV.

## Setup
Grid over `split_mode: [prediction, trajectory]`, everything else exp117/exp120's values. 18 rows:

- **12 suppressed** (seeds 3203–3230 subset) — exp117 rows that screened `no-concept` while plain
  prompt A rendered the object at the same seed. Known 0/12 under `prediction`.
- **6 survivors** (3202, 3216, 3218 `first`; 3209, 3219, 3214 `second`) — exp117 passes spanning
  concept-max 0.848 down to 0.137.

## Reading it
| outcome | means |
|---|---|
| suppressed rows flip AND the 6 survivors still pass | trajectory mode is the yield fix; rebuild the datasets on it and re-run the region-balance count |
| suppressed rows flip but survivors break (seam, or `not-split`) | the shared latent was buying coherence, not just suppression — try merging later (a small joint phase before the tail) |
| nothing flips | the concept region is decided by the initial noise, not by conditioning context; then (prompt, seed) pre-screening is the only lever left and the thread should stop sweeping the sampler |

The `prediction` arm must reproduce 12 fails / 6 passes exactly. If it does not, `resolve_split` is
not giving these rows their exp117 splits and neither arm means anything.

**Screen both arms, then look at them.** `tools/screen_split_dataset.py` now also rejects blank
targets (exp122), but it cannot see a broken seam — check 3-4 clips frame-by-frame at the boundary in
the trajectory arm specifically.

## Status
- [ ] Submitted.
- [ ] `prediction` arm confirmed at 12 fails / 6 passes.
- [ ] Pass counts per arm; seam checked by eye on the trajectory arm.
- [ ] `docs/split_prompt.md` §3.3 updated with the outcome.
