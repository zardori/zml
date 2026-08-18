---
status: done
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  THE YIELD FIX. `split_mode: trajectory` takes exp120's 12 known-suppressed rows from 0/12 to
  12/12 passing, at identical cost, while 5 of the 6 exp117 survivors still pass — 17/18 (94%)
  against the `prediction` control's 6/18 (33%). The control arm reproduced its pre-registered
  12 fails / 6 passes exactly, so the comparison is valid. This confirms exp120's diagnosis: the
  suppression was the shared latent dragging the concept region toward the substitute, not the
  conditioning strength, and removing the coupling beats pushing against it (exp120's gs 9 recovered
  only 7/12 and leaked the concept into the safe half). Seam-by-eye review still pending.
---
# exp127 — splice the trajectories, not the predictions

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

## Results (2026-08-17) — the suppressed rows flip, and the survivors hold

Both arms built 18/18 with zero skips (~1.0 h each on helios). Screened at
`--min-concept-max 0.10`, the thread's standard threshold:

| arm | pass | no-concept | not-split | blank-target | region balance |
|---|---|---|---|---|---|
| `prediction` *(control)* | **6/18 (33%)** | 12 | 0 | 0 | 3 first / 3 second |
| **`trajectory`** | **17/18 (94%)** | 1 | 0 | 0 | 10 first / 7 second |

### The validity check passed exactly

The `prediction` arm returned **12 fails / 6 passes**, and they are the pre-registered rows: p0–p11
(the exp120 suppressed set) all screen `no-concept`, p12–p17 (the exp117 survivors) all pass. So
`resolve_split` gave these rows their exp117 splits and the arms are comparable.

### 0/12 -> 12/12 on the suppressed rows

Every one of exp120's suppressed rows renders the concept under `trajectory`, most of them strongly:

| row | prediction conc_max | **trajectory conc_max** |
|---|---|---|
| p0_s3203 | 0.0406 | **0.5684** |
| p3_s3207 | 0.0102 | **0.5903** |
| p6_s3220 | 0.0957 | **0.8051** |
| p7_s3222 | 0.0911 | **0.8100** |
| p8_s3224 | 0.0003 | **0.7347** |
| p10_s3228 | 0.0097 | **0.5418** |
| p11_s3230 | 0.0107 | **0.4677** |

Contrast indices on the flipped rows are +0.84 to +0.999, i.e. the concept lands in its own half and
the safe half stays clean — which is what exp120's gs 9 could *not* do (5 of its 7 recovered rows
screened `not-split`, the concept leaking into the safe half). Removing the coupling is strictly
better than pushing against it.

This is the outcome row 1 of "Reading it" predicted: **trajectory mode is the yield fix.**

### The one regression

`p13_s3216` goes the other way: conc_max 0.8137 -> 0.0640, a pass that becomes `no-concept`. Its
contrast index stays high (+0.917), so the split still works — the concept simply renders faintly in
a pure-A context at this seed. One row in 18 is a cheap price against +11, but it means trajectory
mode is not a strict superset of prediction and a rebuild should screen, not assume.

Coherence did **not** break: zero `not-split` and zero `blank-target` in either arm, and the region
balance stays workable at 10 first / 7 second. exp076's reading — that the shared initial noise, not
the shared latent, is what buys cross-seam coherence — survives this test.

## Status
- [x] Submitted (grid, 2 jobs, 2026-08-16, ~1.0 h each on helios).
- [x] `prediction` arm confirmed at 12 fails / 6 passes.
- [x] Pass counts per arm: 6/18 vs 17/18.
- [ ] **Seam checked by eye on the trajectory arm** (3-4 clips frame-by-frame at the boundary) — the
      screen cannot see a broken seam, and this is the one failure mode it would miss.
- [ ] `docs/split_prompt.md` §3.3 updated with the outcome.
- [ ] Rebuild the chain-saw and church datasets on `split_mode: trajectory` and re-count yield.
