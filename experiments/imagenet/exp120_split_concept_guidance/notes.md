---
status: ready
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  Sweeps concept_guidance_scale [6, 9, 12] on the 12 exp117 chain-saw rows that failed the screen
  *despite* the plain prompt-A clip rendering the object. exp117 moved the diagnosis from "the model
  never drew it" (fixed by the prompt reframe) to "the splice suppresses a concept the same
  (prompt, seed) renders fine": failing rows keep 6% of the plain-A confidence, passing rows keep
  112%. This is the first knob aimed at that mechanism. Not submitted yet.
---
# exp120 — per-region CFG: can the concept branch hold its half?

## The question
`generate_split_clip` predicts `pred_a` and `pred_b` over the **whole** latent and splices only the
prediction. So `pred_a` is evaluated in a context whose other region is converging on prompt B, and
CogVideoX's temporal-coherence prior — the thing that makes its clips look like one continuous scene
— argues that the concept region should match. Either the object establishes itself early enough to
survive that pull or the substitute swallows it.

That predicts a *binary* failure, and the measurement matches. Over exp117's 30 rows, the split
clip's concept-half mean divided by the same row's plain prompt-A mean:

| rows | ratio (median) |
|---|---|
| passing | 1.12 |
| failing | **0.06** |

There is no middle. The concept half either renders as strongly as an unsplit generation or is gone.

`concept_guidance_scale` raises CFG on `pred_a` only. Cost is zero — both branches' conditional and
unconditional predictions are already computed; this changes a scalar.

## Why these 12 rows
The exp117 rows that screened `no-concept` *and* whose whole-clip prompt-A confidence cleared 0.10 —
i.e. the model demonstrably renders a chain saw for that exact (prompt, seed) when not splitting.
That selection is what makes 12 rows sufficient: under the control arm all 12 are known zeros, so any
pass is signal and no baseline estimation is needed.

Seeds: 3203 3205 3206 3207 3211 3212 3220 3222 3224 3226 3228 3230
(`prompts/imagenet_objects/split/chain_saw_closeup_suppressed.csv`, built by
`tools/subset_split_prompts.py` so the rows are byte-identical to exp117's CSV).

## Predictions, written down first
- **6.0 → 0/12 pass.** Anything else means `resolve_split` is not reproducing exp117's per-row
  `(split_latent_frame, concept_region)` and the whole comparison is void.
- **9.0 → some rows flip.** If the pull from the B context is what loses them, more guidance on the
  A branch is the direct counter.
- **12.0 → more rows flip, quality starts to cost.** High CFG on CogVideoX saturates and over-sharpens.
  A pass bought with a clip too degraded to train on is not a pass.

## Reading it
| outcome | means |
|---|---|
| 9 or 12 clears ≥6/12 at acceptable quality | per-region guidance is the yield lever; fold into exp121/exp122 before they run |
| flips only at 12.0, with visible artefacts | the pull is real but CFG is the wrong instrument — try splicing the *latent* rather than the prediction |
| all three arms 0/12 | the concept region is decided by the initial noise, not by conditioning strength; only (prompt, seed) pre-screening will move yield |

Check the surviving clips by eye as well as by screener — this is the one knob here that can raise
the detector score while making the video worse.

## Status
- [ ] Submitted.
- [ ] Control arm confirmed at 0/12.
- [ ] Pass counts per arm recorded, with a quality check of the clips that flipped.
- [ ] `docs/split_prompt.md` updated with the outcome either way.
