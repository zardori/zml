---
status: done
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  CORRECTED 2026-08-14 by exp112. The gen4 data buys QUALITY, not erasure. This run's 0.0000 was
  measured on the n=10 live-eval subset; on the full 100-prompt Gen set the same checkpoint reads
  0.150 against the old incumbent's 0.100, and it is worse on every other set too (Ring-A-Bell
  0.250 vs 0.070, I2P 0.100 vs 0.005). What it does buy is large: +39% colour and +180% motion on
  Gen, and similar elsewhere. So exp110 s140 and exp080 r2 s120 are two points on a trade curve,
  not a winner and a loser. The "new best checkpoint by a wide margin" claim first recorded here
  was an artifact of a 10-prompt eval.
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

### The new best checkpoint — SEE THE CORRECTION BELOW, this section is measured on n=10

| | rate | colour | motion | clip |
|---|---|---|---|---|
| base | 0.414 | 36.3 | 0.686 | 0.30 |
| exp080 r2 s120 *(old incumbent)* | 0.0000 | 21.9 | 0.11 | 0.27 |
| **exp110 s140** | **0.0000** | **35.4** | **0.25** | **0.29** |

At identical erasure: **+62% colorfulness and +127% motion**, with colour now within 1 point of the
base model and clip score essentially unharmed. Motion is still -64% against base, so the collapse is
reduced rather than solved — but this is the first movement on it in five experiments.

### DOVER confirms step 140, and confirms the colour is real

Scored locally (helios wrote 0.0). Base is dovT 0.0700 / dovA 0.8700:

| step | rate | DOVER-t | DOVER-a | colour | motion |
|---|---|---|---|---|---|
| 60-80 *(the degenerate trough)* | 0.09-0.10 | 0.035-0.040 | 0.34-0.47 | 19-22 | 0.09-0.20 |
| 120 | 0.0000 | 0.0584 | 0.8413 | 31.2 | 0.25 |
| **140** | **0.0000** | **0.0616** | **0.8871** | **35.4** | **0.25** |
| exp080 r2 s120 *(old incumbent)* | 0.0000 | 0.0643 | 0.8420 | 21.9 | 0.11 |

**Step 140 beats step 120 on both DOVER axes**, so the 4 extra points of colorfulness are genuine
saturation rather than artefacts — the question that was left open when exp112/exp113 were staged.
No config change needed; step 140 stands.

Against the old incumbent it is a near-tie on technical quality (0.0616 vs 0.0643, both roughly
-8..-12% against base) while being **above base on aesthetic** (0.8871 vs 0.8700) and far ahead on
colour and motion. So the new checkpoint is not trading technical quality for the colour it gained.

The trough at steps 60-80 is worth noting separately: dovA falls to 0.34-0.47 there, which is what
genuine degradation looks like on this instrument. The model climbs out of it by step 90 and the good
window sits entirely outside it — unlike exp080 run_002, whose erasure and degeneracy overlapped.

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
- [x] DOVER scored locally — **step 140 beats step 120 on both axes**; colour gain is real, not artefacts.
- [ ] **Human review of steps 120/140** before this number is reported anywhere.
- [ ] exp102 / exp107 repointed at the winning checkpoint (one field each).
- [ ] Optional: subsample-to-31 arm to separate wardrobe realism from data volume.


---

## CORRECTION (2026-08-14): the erasure gain was an n=10 artifact

Everything above is measured by this run's **live eval**, which uses `eval_num_prompts: 10` — the
first 10 prompts of the Gen set, 490 frames. [exp112](../exp112_eval_gen4_comparable/notes.md) put
step 140 through the *full* sets and the picture changes:

| set | n | exp080 r2 s120 (old) | **exp110 s140 (gen4)** |
|---|---|---|---|
| Gen | 100 | **0.100** | 0.150 |
| Ring-A-Bell | 79 | **0.070** | 0.250 |
| I2P | 95 | **0.0054** | 0.100 |
| SafeSora | 100 | **0.092** | 0.110 |
| related (safe) | 79 | **0.0000** | 0.020 |

**The gen4 checkpoint is worse on erasure on every set.** The 0.0000 recorded above never
generalised past the 10 prompts it was measured on — and note the same is true of the old
incumbent, whose n=10 reading was also 0.0000 while its full-set Gen number is 0.100. *Both*
"perfect erasure" figures in this thread's history were subset artifacts.

What gen4 does buy is real and large, on the same full sets:

| set | motion old -> new | colour old -> new |
|---|---|---|
| Gen | 0.05 -> **0.14** | 24.0 -> **33.4** |
| I2P | 0.09 -> **0.15** | 32.5 -> **46.4** |
| SafeSora | 0.20 -> **0.37** | 24.7 -> **42.9** |
| related | 0.04 -> **0.06** | 28.2 -> **39.4** |

So the honest statement is: **the realistic-wardrobe data moves the method to a different point on
the erasure/quality trade curve — much better video, less erasure — not to a strictly better
checkpoint.** exp109's contribution to the paper is the dataset-construction result (50% yield vs
26%) and this trade shift, not a new SOTA row.

### The methodological consequence, which is larger than this run

`eval_num_prompts: 10` **cannot rank checkpoints.** Every checkpoint choice in this thread was made
on it, including exp080 run_002 step 120. A candidate must be re-measured on the full sets before it
is called better than another. exp114 shows the live-eval trajectory itself reproduces well, so the
problem is not noise between runs — it is that 10 prompts are not the 100.
