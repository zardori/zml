---
status: done
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  THE DATA MATTERED AFTER ALL, and this is the new best checkpoint by a wide margin. Step 140 reads
  rate 0.0000 at colorfulness 35.4 (base 36.3 — essentially no colour loss) and motion 0.25, against
  the old incumbent exp080 r2 s120's 0.0000 / 21.9 / 0.11. That is +62% colour and +127% motion at
  identical erasure. The window is wide, not a transient: steps 90-140 are all <=0.04 with four
  checkpoints <=0.01. The U-shape also flattens — the rebound tops out at 0.23 where every earlier
  run reached 0.49-0.76. Overturns exp088's "data does not matter" reading. Needs DOVER + human
  review, then exp102/exp107 repointed.
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

## Results (2026-08-12) — the data mattered

| step | 60 | 70 | 80 | **90** | **100** | 110 | **120** | 130 | **140** | 150 | 160 | 170 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rate | 0.09 | 0.10 | 0.09 | **0.01** | **0.01** | 0.04 | **0.00** | 0.04 | **0.00** | 0.21 | 0.06 | 0.23 | 0.12 |
| colour | 22.3 | 19.4 | 19.3 | 26.2 | 28.3 | 30.8 | 31.2 | 33.6 | **35.4** | 37.0 | 36.1 | 36.8 | 35.7 |
| motion | 0.20 | 0.09 | 0.11 | 0.22 | 0.22 | 0.24 | 0.25 | 0.27 | **0.25** | 0.30 | 0.29 | 0.21 | 0.13 |
| clip | 0.30 | 0.26 | 0.28 | 0.30 | 0.27 | 0.28 | 0.29 | 0.30 | 0.29 | 0.28 | 0.28 | 0.28 | 0.27 |

### The new best checkpoint

| | rate | colour | motion | clip |
|---|---|---|---|---|
| base | 0.414 | 36.3 | 0.686 | 0.30 |
| exp080 r2 s120 *(old incumbent)* | 0.0000 | 21.9 | 0.11 | 0.27 |
| **exp110 s140** | **0.0000** | **35.4** | **0.25** | **0.29** |

At identical erasure: **+62% colorfulness and +127% motion**, with colour now within 1 point of the
base model and clip score essentially unharmed. Motion is still -64% against base, so the collapse is
reduced rather than solved — but this is the first movement on it in five experiments.

### It is a window, not a transient

Steps 90-140 are all <=0.04, with **four checkpoints at <=0.01** (90, 100, 120, 140). That matters
because isolated single-step zeros have been misread as regimes three times in this thread. Six
consecutive checkpoints in the low band is a regime.

### The U-shape flattens

Every earlier run rebounded to 0.49-0.76 as colour recovered. This one tops out at **0.23** (step
170), and the late window (160-200) aggregates to **rate 0.124 at motion 0.212 and colour 36.2** —
i.e. at *full* colour recovery it still holds a 70% reduction against base. Compare exp086 r3's late
window at 0.506, exp088 r1's 0.440, exp105 r2's 0.290.

### This overturns exp088

[exp088](../exp088_frame_replace_nudity_clean/notes.md) unfroze the donors, changed nothing, and its
recorded conclusion was that the method is insensitive to its training data. That reading was wrong —
or rather, too broad. Un-freezing donors did nothing; **replacing implausible wardrobe with realistic
wardrobe (and 34 targets with 100) did a great deal.** The distinction is that exp087 fixed a
*mechanical* defect in the targets while exp109 fixed a *semantic* one.

The volume confound stated in the config header stands and cannot be resolved from this run: 100
targets vs 34, so realism and data volume moved together. A subsample-to-31 arm would separate them.
Worth one job now that the result is positive, where it was not worth one when the expected outcome
was a null.

## Status
- [x] exp109 reviewed (100/200) and merged into `combined_dataset/` on the cluster.
- [x] Submitted and complete (1 job, 200 steps).
- [x] Read against exp080 run_002 step 120 — **beats it on every axis at equal erasure**.
- [ ] DOVER scored locally (helios wrote 0.0) — needs the eval videos pulled.
- [ ] **Human review of steps 120/140** before this number is reported anywhere.
- [ ] exp102 / exp107 repointed at the winning checkpoint (one field each).
- [ ] Optional: subsample-to-31 arm to separate wardrobe realism from data volume.
