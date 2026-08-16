---
status: done
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  Split verdict: the MECHANISM is confirmed, the KNOB is rejected. Pass counts on the 12 suppressed
  rows are 0/12 (gs 6, control, exactly as predicted) -> 2/12 (gs 9) -> 3/12 (gs 12), well under the
  pre-registered >=6/12 gate, so concept_guidance_scale is not the yield lever and must NOT be folded
  into new builds. But at gs 9 SEVEN of the 12 rows render the concept again (2 pass + 5 not-split)
  where zero did at gs 6 — raising CFG on the A branch does defeat the suppression, and the concept
  then leaks into the safe half, which is a different failure. That localizes the problem to the
  shared latent context and prescribes exp124's dual-trajectory splice.
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

## Results (`grid_20260816_001926`, helios, ~1.6 h per arm)

Screened with `tools/screen_split_dataset.py` at the standard 0.10 / 0.4 gates.

| guidance | pass | not-split | no-concept | renders the concept at all |
|---|---|---|---|---|
| 6.0 (control) | **0/12** | 0 | 12 | 0/12 |
| 9.0 | 2/12 | 5 | 5 | **7/12** |
| 12.0 | 3/12 | 1 | 8 | 4/12 |

**The control arm reproduced 0/12 exactly**, which was the pre-registered validity check: `resolve_split`
does give each row exp117's `(split_latent_frame, concept_region)`, so the comparison stands.

**The gate (≥6/12) is not met — do not fold `concept_guidance_scale` into new builds.** As a yield
lever it buys at most 3 rows out of 12, i.e. roughly +8 percentage points on a 30-row build, for a
knob that also has to be swept.

**But the mechanism in `docs/split_prompt.md` §3.3 is confirmed.** Read the last column, not the pass
column: at gs 9, seven of twelve rows render the chain saw where zero did at gs 6. The suppression is
real and CFG on the A branch does defeat it. What then fails is *localization* — five of those seven
land in `not-split`, meaning the safe half now reads as much chain saw as the concept half
(p0_s3203: 0.471 vs 0.470; p5_s3212: 0.645 vs 0.441). The concept comes back and immediately bleeds
across the seam.

**Per-row behaviour is not monotone in guidance**, so part of what gs changes is which sample you get,
not how strongly the concept is drawn: p2_s3206 reads concept-max 0.633 at gs 9 and 0.0004 at gs 12;
p5_s3212 goes 0.783 → 0.0019. Only s3220 and s3228 pass at both 9 and 12. A knob whose effect on a
given row flips sign between settings cannot be tuned into a reliable builder.

**Where this points.** Both findings — the concept returns under stronger A-guidance, and it then
contaminates the B half — say the two regions are coupled through the *shared latent*: `pred_a` and
`pred_b` are computed on one tensor and only the prediction is spliced. That is exactly the third row
of the reading table above ("CFG is the wrong instrument"), and its prescription is exp124: give each
branch its own trajectory during the split phase and splice once, so the concept region is denoised in
a pure-A context and the safe region never sees prompt A at all.

## Status
- [x] Submitted (2026-08-16, helios, 3 jobs).
- [x] Control arm confirmed at 0/12.
- [x] Pass counts per arm recorded.
- [x] `docs/split_prompt.md` §3.3 and `docs/imagenet_objects.md` §5 updated.
- [ ] Frame-level quality check of the 3 clips that flipped at gs 12 (screener cannot see saturation;
      the not-split rows make the quality question moot for gs 9).
