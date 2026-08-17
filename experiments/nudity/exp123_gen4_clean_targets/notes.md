---
status: active
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  Tests why 34 distorted targets erase better than 100 realistic ones. Key measurement: the gen4
  training targets themselves read 0.198 tagged-frame fraction on the eval's own detector (old
  dataset: 0.001), and exp114 converges to exactly that band — the run cannot score below the
  detector response of its own supervision. 75/100 gen4 targets are clean (floor 0.0016); this
  trains on those 75 at eta [2.0, 3.0], separating the target-floor mechanism from the
  donor-distance mechanism in one grid. 2 jobs.
---
# exp123 — clean gen4 targets, eta grid

## The mechanism question

exp112 established that gen4 (100 realistic targets) erases worse than the old 34 distorted ones at
identical hyperparameters, and exp114 established that longer training does not close the gap even
at matched per-example exposure (~20 visits/example). Three candidate mechanisms, one measurement
each:

### A. The supervision carries the concept, as the instrument sees it
Computed locally from `edited_frame_confidences` in exp109's filtered metadata:

| training set | tagged-frame fraction (>=0.2) |
|---|---|
| old clean 31 (exp080) | **0.001** |
| gen4 kept 100 (exp109) | **0.198** |
| gen4 clean-75 subset | **0.0016** |

**exp114's late phase converges to 0.19-0.27 — the detector response of its own supervision.** The
model is faithfully learning targets one-fifth of whose frames the eval instrument already reads as
nude. Whether those are fitted-fabric false positives (human review says the clips look fine) does
not matter to the loss: SFT pulls generations toward those latents, and what fires NudeNet in the
targets fires NudeNet in the outputs. **You cannot erase below the floor of what you train toward.**

Concentrated, fixably: 25 targets carry nearly all of it (worst categories: casual 0.335,
formal_wear 0.312; best: traditional 0.064, outerwear 0.067). The 75 targets at <=5% form a set
whose floor matches the old dataset's.

### B. The erase push scales with the donor gap
`target - teacher = eta * (donor - teacher)`. Old wardrobe: shapeless dark sacks, donors far from
the nude teacher — a large, coherent push. gen4: fitted clothing that deliberately keeps the shot
and changes only the wardrobe — donors close to the teacher, so eta=2 (tuned on the old data's
gaps, never re-tuned) delivers a much smaller displacement. The "parameter that prevents it on the
larger dataset" may literally be eta.

### C. A rank-8 LoRA learns the shared component, and gen4 diversified it away
Old data: one wardrobe -> shared component of (donor - teacher) IS "replace skin with cloth" ->
generalizes. gen4: 8 wardrobe categories and varied colour by design -> clothing directions
partially cancel; the surviving shared component is *style*. Direct evidence the model absorbed
exactly that: exp113 measured colorfulness ABOVE base on unrelated sets (53.1 vs 45.8, 49.6 vs
40.5). The example-specific clothing residual is memorized, not generalized — which is what a
smoothly falling `train/loss_erase` (exp114's curve) alongside flat eval erasure means.

## Design
One dataset change (clean-75), one grid (eta [2.0, 3.0]):

- **run_001, eta 2.0** — isolates A. Same push as exp110, floor removed.
- **run_002, eta 3.0** — A + B compensated. If only this arm recovers, the donor gap was the
  binding constraint; if both recover, the floor was.
- If neither recovers, C dominates -> next arm is a homogeneous low-floor subset
  (outerwear+traditional, 26 clean targets), which de-diversifies without reintroducing sacks.

`eval_num_prompts: 25` (not 10): n=10 produced two false "0.0000" checkpoints (exp080's and
exp110's; both read 0.10-0.15 on full sets). 1225 frames per point at save_interval 20 keeps wall
clock near exp110's. Any winner still gets the full exp112 treatment before any claim.

## Cluster-side merge (once, before submitting)
```
uv run python zml/precompute/merge_frame_replace_datasets.py \
  --source experiments/nudity/exp109_split_nudity_gen4_dataset/metadata_human_filtered_clean_run001.json experiments/nudity/exp109_split_nudity_gen4_dataset/grid_20260811_134439/run_001/outputs/latents \
  --source experiments/nudity/exp109_split_nudity_gen4_dataset/metadata_human_filtered_clean_run002.json experiments/nudity/exp109_split_nudity_gen4_dataset/grid_20260811_134439/run_002/outputs/latents \
  --source experiments/nudity/exp109_split_nudity_gen4_dataset/metadata_human_filtered_clean_run003.json experiments/nudity/exp109_split_nudity_gen4_dataset/grid_20260811_134439/run_003/outputs/latents \
  --source experiments/nudity/exp109_split_nudity_gen4_dataset/metadata_human_filtered_clean_run004.json experiments/nudity/exp109_split_nudity_gen4_dataset/grid_20260811_134439/run_004/outputs/latents \
  --output_dir experiments/nudity/exp109_split_nudity_gen4_dataset/combined_dataset_clean75
```
Should print `Merged 4 sources -> 75 targets`.

## Ruled out already
- Per-example exposure: exp114 matched the old per-example visit count; no recovery.
- Run-to-run noise: exp114 reproduces exp110 within 0.01-0.04 at every shared checkpoint.
- Retention, lr, rank, timesteps: identical across both datasets.

## Status
- [x] Clean-75 subset built (per-run metadata at exp109 root, committed).
- [ ] Cluster merge run; submitted (2 jobs).
- [ ] Arms read against exp110 (n=10 caveat) and, for any winner, the full exp112 battery.
- [ ] If neither arm recovers: homogeneous outerwear+traditional arm.


## Partial results (2026-08-16, through step 160 of 200)

Calibration first: exp123 evaluates on the first 25 Gen prompts. Rescoring the two existing
checkpoints' saved exp102/exp112 clips on **exactly that subset** puts everyone on one scale:

| checkpoint | first-25 Gen rate |
|---|---|
| old (exp080 r2 s120) | 0.1200 |
| gen4-100 (exp110 s140) | 0.1233 |

### Arm r1 (clean-75, eta 2.0) — mechanism A alone: refuted for this regime, possibly inverted

| step | 60 | 80 | 100 | 120 | 140 | 160 |
|---|---|---|---|---|---|---|
| rate | 0.08 | 0.07 | 0.14 | 0.23 | **0.26** | 0.26 |
| colour | 21.3 | 22.6 | 27.3 | 34.0 | 35.2 | 37.9 |

At step 140, clean-75 reads 0.26 against exp110's 0.123 on the same subset — which was first
recorded here as "removing the detector-visible targets made erasure worse".

**CORRECTED (2026-08-16): that comparison had a phase confound.** With 75 targets each example is
visited 4/3 as often per step, and the U-shape tracks per-example visits, not steps: matched-step
comparison put this run deep in its rebound (7.5 visits/example) against exp110 at its trough
(5.6). At matched visits (~5.5), clean-75 reads 0.14 against full-100's 0.123 — parity within
single-run noise. So the floor fix neither helped nor hurt at eta 2 in this window; the floor
remains the explanation for exp114's converged limit (which sits exactly at 0.198), and the
clean-75 is kept as the base dataset going forward: same dynamics, supervision floor 0.0016
instead of 0.198. The "dirty-25 as boundary supervision" speculation recorded here previously
loses its supporting datapoint and is withdrawn.

### Arm r2 (clean-75, eta 3.0) — mechanism B: supported, and it is the knob that works

| step | 60 | 80 | 100 | 120 | 140 | 160 |
|---|---|---|---|---|---|---|
| rate | 0.03 | 0.12 | 0.00* | 0.12 | **0.11** | 0.17 |
| colour | 16.9 | 17.0 | 21.5 | 28.4 | 30.7 | 35.5 |
| motion | 0.04 | 0.03 | 0.06 | 0.11 | 0.10 | 0.11 |

Same data, same eval, only eta: **eta 3 beats eta 2 at every step past 40**, and at matched colour
(~35) reads 0.17 vs 0.26. The erase push is `eta * (donor - teacher)`; fitted donors shrank the gap
and raising eta compensates. Clip score holds 0.27-0.29 throughout, so text conditioning survives
eta 3.

*The step-100 0.0000 (3/1225 frames) sits in the degenerate trough (colour 21.5) between two 0.12
neighbours — an isolated dip even at n=25, per the standing rule not a checkpoint.

### Where r2 s140 lands
rate 0.11 vs old 0.120 / gen4-100 0.123 on the same subset — erasure parity with both, quality
between them (colour 30.7, motion 0.10). An interpolation on the existing frontier, not a
breakthrough — but the eta trend is monotonic and untested past 3.0, and r1's result says the
follow-up should run on the FULL 100, not the clean subset. That is exp124.

## Status
- [x] Cluster merge run (75 targets); submitted (2 jobs).
- [x] Partial read at n=25 with subset-calibrated baselines.
- [ ] Final pull (steps 180-200).
- [ ] exp124 (full-100, higher eta) — the follow-up both arms point at.
