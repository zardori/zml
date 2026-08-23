---
status: ready
concept: nudity
method: frame_replace_split
thread: nudity
takeaway: >
  READY, NOT SUBMITTED. gen5 dataset build: gen4's 25 scenes with a deliberately UNIFORM wardrobe
  (3 near-identical navy pieces vs gen4's 40 varied garments), 200 triples, seeds 5601-5800. Tests
  whether donor-wardrobe homogeneity raises edit coherence enough to change what the LoRA learns.
  Gated: measure coherence with tools/analyze_edit_directions.py before submitting exp146 —
  reference values OLD-31 0.714 (n=34), GEN4-100 0.605, CLEAN-75 0.615.
---
# exp145 — gen5 dataset, uniform wardrobe

## Why
exp080's 34-clip gen1-gen3 set still yields the best checkpoint, beating exp110/exp123/exp124/exp136
on the realistic gen4 data. `tools/analyze_edit_directions.py` was written to explain that and came
back **negative**: over every seed, gen4's edits are *larger* (13.2 vs 9.9), the shared component is
the same size (3.45 vs 3.38), chroma:luma is identical (~0.45), and the coherence gap (0.605 vs
0.714) is inside the bootstrap spread of 34-clip gen4 subsets — 21% of random subsets reach 0.714,
n=2000. An earlier "old data is chroma-dominated" claim was an artifact of a partial seed match and
is retracted (`docs/frame_replace.md`, commit 1be8127).

So this build is **not** implied by a measured finding. It is a deliberate manipulation of the one
variable with a mechanistic story attached — a rank-r LoRA can only realise the component of
`donor - teacher` that recurs across examples, and gen4 was built to maximise variety — run at a
size where the coherence difference cannot be sampling noise.

## Setup
`prompts/split_nudity_gen5.csv`, built by `tools/build_split_nudity_gen5_prompts.py`, which
**imports** gen4's `SCENES`, `Row`, `_write` and banned-wardrobe list rather than restating them.
Only `GARMENTS` differs:

| | gen4 (exp109) | gen5 (this) |
|---|---|---|
| garments | 40, across 8 categories | **3, all navy, one silhouette** |
| distinct B prompts | 200 | **75** |
| scenes | 25 | 25 (same) |
| rows / seeds | 200 / 3801-4000 | 200 / **5601-5800** |
| split geometry | 0.85, jitter 2, frame 7 | identical |

Three garments and not one: a single garment is the cleanest manipulation but the worst collateral
risk — the adapter could learn "render this exact shirt". Three pieces in one colour family keep
coherence near-maximal while leaving some within-concept variation. `verify()` asserts every garment
carries the `navy` token, so a later edit cannot quietly reintroduce colour variety.

Navy was chosen against the two failure modes earlier generations hit — gen1-gen3's cream sacks read
as skin, and read as bulky — not from colour theory. The chroma rationale is retracted.

## Gate
This experiment produces data, not a result. Before exp146 is submitted:

1. `tools/screen_split_dataset.py` as for any build — the within-clip differential, and whether the
   base model rendered the concept at all under prompt A.
2. `tools/analyze_edit_directions.py --metadata <filtered> --videos <dir> --label GEN5`.
   **If coherence does not land clearly above OLD-31's 0.714 at n>=75, the manipulation failed and
   exp146 should not be submitted.** Record the number here either way — a null result here is
   worth as much as a positive one, because it closes the coherence hypothesis.
3. Human review of the edited clips for residual concept (`edited_max_confidence` first pass), the
   defect exp109's notes flag in 4 of exp080's 34 targets.

## Status
Not submitted. Prompt CSV built and committed; config staged.
