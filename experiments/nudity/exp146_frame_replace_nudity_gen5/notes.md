---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  exp145's GATE PASSED — edits on gen5 are 2x more directionally aligned than the incumbent's
  (pairwise cosine 0.609 vs 0.293) and 4x gen4's. Trains exp110's exact regime on gen5 instead of
  gen4, so donor-wardrobe diversity is the only variable, except that eta is now swept [1.0, 2.0]
  because gen5's edits are 2-3x larger and eta 2.0 alone risks burning the window. Still blocked on
  human review of exp145's 57 screened clips and the cluster-side merge.
---
# exp146 — frame_replace on gen5 (uniform wardrobe)

## Why
The training arm of exp145. See that experiment's notes for why a uniform wardrobe was worth testing
even though the LAB edit statistics came back negative: it is a deliberate manipulation of the one
variable with a mechanistic story, not a finding being followed up.

**The manipulation worked at the data level.** On exp145's screened 57, mean pairwise cosine between
clip edit directions is 0.609, against 0.293 for OLD-31 and 0.154/0.149 for GEN4-100/CLEAN-75. Read
that statistic rather than coherence: coherence is magnitude-weighted, reads a misleading 0.729 on
this set, and is the same small-n-inflated number that made OLD-31's 0.714 unreliable. Yield was
57/200 against gen4's 63/200 under an identical rule, so uniformity cost nothing.

## What is held fixed
Everything except the dataset. `config.yaml` copies exp110's training fields verbatim — eta 2.0,
rank 8, alpha 8, lr 1e-4 constant, 200 steps, `save_interval` 10, grad-accum 4, velocity loss on
original latents, timesteps 400-1000, `retention_weight` 1.0 against exp041's fire anchors, seed 42.
The comparison is exp146 vs exp110, same regime, different donor diversity.

**One departure that does touch training: `erase_esd_eta: [1.0, 2.0]`.** gen5's edits are 2-3x
larger than anything trained on before (LAB magnitude 28.3 vs OLD-31's 9.9). The erase push is
`eta * (donor - teacher)`, so eta 2.0 on this data is a much bigger step than eta 2.0 on gen4's, and
the run could pass its useful window long before step 120 — exp125 burned a run in exactly this way
when effective step size scaled without anyone noticing. eta 2.0 preserves the literal comparison
against exp110; eta 1.0 is the arm matched to the larger displacement. The proxy caveat matters:
magnitude is measured in decoded LAB pixels, training happens in latent/velocity space.

Two further departures, neither touching training:
- **`eval_num_prompts: 25`, not exp110's 10.** Ten videos cannot resolve a detection rate; exp110's
  two "0.0000" checkpoints were an artifact of it and were reported as a new best before the full
  sets contradicted them.
- **`control_related_prompts` / `control_unrelated_prompts` added.** A uniform donor set is the
  configuration most likely to teach "render this navy shirt" rather than "do not render nudity",
  and exp113 measured the gen4 checkpoint shifting colour on nudity-free prompts (53.1 vs base
  45.8). Without these sets that collateral is invisible until a separate eval job.

## What counts as success
Not a lower nudity rate on its own. gen4-derived runs already erase *deeper* than the incumbent
(exp123 r1 s80 = 0.070, exp136 r1 s200 = 0.040, vs exp080 r2 s120 = 0.120) — they just only do it
inside the degeneracy trough and hand it back as sharpness returns. The thing exp080 has and they do
not is a checkpoint that is erased **and** sharp simultaneously.

So the read is: **is there a checkpoint with a low `nudity_frame_rate` at DOVER-t >= 0.058?** For
reference, at each run's first checkpoint clearing that bar the frame rate is 0.000 for exp080 s120
and exp110 s120-140, but 0.23 for exp123 r1 and exp136 r1 and 0.32 for exp123 r2. DOVER reads 0.0 on
helios (aarch64) — score post-hoc with `tools/score_dover.py`, and never read colorfulness as
quality.

## The dataset: 27 hand-picked clips from shard 1

Shards 2-4 will not be reviewed, so this arm is shard-1-only: **27 clips**, against OLD-31's 34 —
the size that produced the incumbent. `metadata_human_filtered_run_001.json` at exp145's root.

**17 clean, 10 partial**, and the partial ones are there on purpose. They are clips where the
concept is *partially* hidden — nude but turned away, or skin blending into the garment. The
detector cannot distinguish "the graft failed and left bare skin" from "this is an intermediate
skin/cloth state", but a reviewer can, and the second is exactly the state the model passes through
during unlearning. Each entry carries a `review_class` field (`clean` / `partial`) so this split is
usable later without re-reviewing.

The objection I raised against admitting them — that residual concept in the target caps erasure —
does not survive checking. It rested on "old data floor 0.001", which is **clean-75's** number
mis-attributed. Recomputed:

| dataset | mean frame conf of target | mean clip max | outcome |
|---|---|---|---|
| CLEAN-75 | 0.001 | 0.013 | worst |
| OLD-31 | 0.044 | 0.062 | **best checkpoint** |
| GEN4-100 | 0.089 | 0.129 | middle (0.150) |

The *cleanest* dataset performed worst. Target cleanliness does not predict outcome in any of the
three datasets we have, so it is not a reason to drop the partial clips.

## Curriculum — a hypothesis this dataset makes testable, not a claim
If partial targets are supervision for the states unlearning actually visits, then ordering them
against clean ones should matter: partial-first (start where the model already is) or partial-last
(corrective fine-tuning once it stalls). That is a real technique — covering the induced state
distribution rather than only the endpoint — and `review_class` is the field a curriculum run would
sort on. Not staged: it is only worth building after exp146/exp147 show gen5 does something at all.

## Status
Not submitted; blocked on the cluster-side merge.
