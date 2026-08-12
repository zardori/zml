---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  exp080 run_002's exact regime (fire retention, eta 2.0, lr 1e-4) on exp109's 100 realistic-wardrobe
  targets — one variable against the best checkpoint this project has. Deliberately NOT paired with
  clothed retention: exp108 shows that trade is monotonic, so pairing would confound two changes for
  no gain. Confound that IS unavoidable and stated: 100 targets vs 34, so data volume moves with
  wardrobe realism. 1 job.
---
# exp110 — frame_replace on the gen4 dataset

## Why

[exp109](../exp109_split_nudity_gen4_dataset/notes.md) rebuilt the training data after human review
found gen1-gen3's edited clothing implausibly baggy or skin-toned. The cause was prompt design: every
`prompt_b` was written to satisfy "no bare skin visible", and the cheapest way to satisfy that is a
shapeless sack. gen4 names fitted, ordinary garments instead and bans bulk and skin-adjacent wardrobe
at build time. Review kept **100 of 200 (50%)** against exp078's 26%.

This run asks the only question that matters about that: **does better training data produce a better
checkpoint?** [exp088](../exp088_frame_replace_nudity_clean/notes.md) already answered "no" for one
data axis — un-freezing the donors changed nothing — so this is not a foregone conclusion, and a null
here would be a genuinely useful second data point about how insensitive this pipeline is to its
targets.

## One variable

Everything is exp080 run_002 — fire retention at 1.0, eta 2.0, lr 1e-4, 200 steps, concept-only eval
— except `metadata_file` / `latents_dir`. exp080 run_002 step 120 (rate 0.0000, colour 21.9, motion
0.11) is the incumbent, so it is the baseline this reads against.

`save_interval` is 10 rather than exp080's 20, matching every arm since exp086 so trajectories
overlay directly. That is a resolution change, not a regime change.

### Why not exp108's retention weight

The original plan was to pair gen4 with whatever `retention_weight` exp108 picked. exp108 has now
made that pointless. At step 110/200 the sweep shows the trade is **monotonic** — the weight that
restores erasure hands back exactly the motion protection it bought:

| | first rate <=0.02 | motion there |
|---|---|---|
| w0.25 | step 60 | 0.10 |
| w0.5 | step 60 | 0.14 |
| w1.0 (exp105 r2) | never reaches it | (0.13-0.21) |
| fire (exp080 r2) | step 80 | 0.14 |

w0.5's best point and fire's best point are the same point. Pairing gen4 with clothed retention would
change two things and gain nothing. If exp108's late phase (steps 160-200, where exp105 was
interesting) contradicts this, adding a clothed arm is one field on top of this config.

## The confound that cannot be removed

gen4 gives **100 targets where exp080 had 34** (31 after filtering the contaminated ones). So this
run changes wardrobe realism *and* triples the data. Controlling for it would mean subsampling gen4
to 31, which spends a job to answer a question that does not change what we report — more data is
not a finding. Stated here so it is not discovered in review.

## What to watch

- **Against exp080 run_002 step 120: rate 0.0000, colour 21.9, motion 0.11.** A better checkpoint has
  to reach a comparable rate at higher colour or motion.
- **Whether the U-shape flattens.** Every run on the old data erases hardest where the model is most
  degraded and rebounds as colour recovers. If richer, more realistic targets are a stronger learning
  signal, the erasure window should widen or the rebound weaken. If the U is identical, the data
  genuinely does not matter to this method — which, with exp088, would be the finding.
- **Two adjacent checkpoints, never one.** Isolated single-step zeros have been misread as regimes
  three times in this thread and contradicted by their neighbours every time.
- **The late window (160-200) aggregated.** 5 checkpoints is 2450 frames and far better powered than
  any single n=10 read; it is where exp105's real behaviour showed up.
- Per [[feedback-detector-metrics-not-ground-truth]], human review before any number is reported.

## Downstream
If this wins, it becomes the method row: repoint
[exp102](../exp102_eval_frame_replace_comparable_nudity/notes.md) and
[exp107](../exp107_vbench_utility_frame_replace/notes.md) at the new checkpoint — one field each, so
the comparisons survive the swap. A **combined** arm (gen4's 100 + exp080's filtered 31 = 131) is
worth running for the headline number regardless of this result, but it is a data-volume run, not an
ablation, and should not be confused for one.

## Status
- [x] exp109 reviewed (100/200) and merged into `combined_dataset/` on the cluster.
- [ ] Submitted (1 job).
- [ ] Read against exp080 run_002 step 120 and against exp088 (the other data-axis null).
- [ ] Human review of the best checkpoint.
