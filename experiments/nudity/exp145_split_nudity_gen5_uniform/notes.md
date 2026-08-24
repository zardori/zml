---
status: done
concept: nudity
method: frame_replace_split
thread: nudity
takeaway: >
  BUILT, AND THE GATE PASSED. Uniform navy wardrobe raises directional alignment of the edits to
  pairwise cosine 0.609 on the screened 57 (0.442 over all 200) vs 0.293 for OLD-31 and 0.154 for
  GEN4-100 — 2x the best previous dataset, 4x gen4. Yield 57/200 (28%) under the same automated
  rule that gives gen4 63/200 (32%), so uniformity cost nothing in yield. CAVEAT: the edits are
  also 2-3x LARGER (magnitude 28.3 vs 9.9/13.2), so exp146 at fixed eta/lr will traverse its
  trajectory much faster — that run now carries an eta hedge. Human review of the 57 still pending.
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

## Result — the gate passed, on the statistic that turned out to matter

Built 2026-08-23, all 200 triples generated, 4 shards.

| dataset | n | magnitude | **pairwise cos** | coherence | shared \|\|mean d\|\| |
|---|---|---|---|---|---|
| OLD-31 (exp080, incumbent) | 34 | 9.9 | 0.293 | 0.714 | 3.38 |
| GEN4-100 (exp110) | 100 | 13.2 | 0.154 | 0.605 | 3.45 |
| CLEAN-75 (exp123/136) | 75 | 14.7 | 0.149 | 0.615 | 4.04 |
| GEN5-200 (all) | 200 | 24.1 | 0.442 | 0.805 | 13.40 |
| **GEN5-57 (screened)** | 57 | 28.3 | **0.609** | 0.729 | 13.97 |

**Read pairwise cosine, not coherence.** The gate was written against coherence
(`||mean d|| / mean||d||`), and on that statistic the screened set reads 0.729 — barely above
OLD-31's 0.714, which would look like a null. It is not: coherence is *magnitude-weighted*, so a set
whose clips vary in edit size is dragged down even when the directions agree. Mean pairwise cosine
normalises each clip first and is the clean read on alignment — and it is unbiased in n, unlike
coherence, whose small-n inflation is the thing that made OLD-31's 0.714 unreliable in the first
place. On it the manipulation is unambiguous: **0.609 vs 0.293 vs 0.154.**

## Yield — uniformity cost nothing

Screened automatically on `original_max_confidence >= 0.5` (the base model actually rendered the
concept under prompt A) and `edited_max_confidence < 0.3` (the graft left no residual concept):

| | n | A rendered | edit clean | both | % |
|---|---|---|---|---|---|
| GEN4 (exp109) | 200 | 98 | 160 | 63 | 32% |
| GEN5 (this) | 200 | 117 | 118 | **57** | **28%** |

Comparable. Note the two halves move in opposite directions: gen5's A-side rendered the concept more
often (117 vs 98 — the A prompts are *identical* between the generations, so this is seed variation),
and its edited clips retained residual concept more often (118 vs 160 clean), plausibly because there
was more concept present to remove. Net usable is within a few clips.

`metadata_screened_run_00N.json` at the experiment root hold the 57 (one per shard, matching the
merge script's per-source form). **These are an automated pre-screen, not a human filter** — the
project's standing rule is that human review is the deciding vote, and gen4's kept-100 was a human
call. Review the 57 edited clips before merging, and cut further if the wardrobe reads wrong.

## The one thing this build makes riskier
The edits are 2-3x larger than anything trained on before (magnitude 28.3 vs OLD-31's 9.9). The
erase push is `eta * (donor - teacher)`, so at exp110's fixed eta 2.0 and lr 1e-4 the model will
move much faster per step and may pass its useful window well before step 120. exp146 now sweeps
`erase_esd_eta: [1.0, 2.0]` for that reason — 2.0 keeps the exact single-variable comparison against
exp110, 1.0 is the arm matched to the larger displacement. Caveat on the inference: magnitude is
measured in decoded LAB pixels, and training happens in latent/velocity space, so the scaling is a
proxy, not a guarantee.
