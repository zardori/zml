---
status: done
concept: nudity
method: eval
thread: nudity
takeaway: >
  Base-model reference on two EXTERNAL nudity benchmarks. Both pass the gate: I2P 0.326 (n=95),
  SafeSora 0.480 (n=100), unrelated 0.000. First nudity eval with usable power (in-house set is
  n=10, where one clip is 10pp). Erasure claims should be made here, not on cogvideox_nudity.csv.
---
# exp082 — base-model reference on external nudity benchmarks

## Why
Verified 2026-08-07: `prompts/cogvideox_nudity.csv`, the set every nudity result so far is measured
on, has **zero** prompts in common with the actual I2P release — it is I2P-*styled* text we wrote,
not the benchmark. That blocks the paper twice over: our numbers can't be tabled next to published
work that reports on real benchmarks, and having authored both the training and eval prompts
invites the objection that a 0.0 detection rate is partly vocabulary overlap.

This is the reference ("Original") row on sets nobody on the team wrote. Provenance, filters and
licences: **`docs/external_eval_sets.md`**.

## Setup
`mode: eval`, no `lora_checkpoint_dir` -> unmodified base model. `control_concept_prompts` is a
list, so `submit_job.py` grids it into one job per benchmark:

- **run_001 — `prompts/i2p_nudity.csv`** (95 prompts). Real I2P, filtered to
  `nudity_percentage > 50` (a column of the benchmark: measured fraction of reference SD-1.4
  samples containing nudity). Uses I2P's own `evaluation_seed` per prompt, so the seed policy is
  satisfied by the benchmark rather than by us.
- **run_002 — `prompts/safesora_nudity.csv`** (100 prompts). SafeSora (CC-BY-NC-4.0), a
  text-to-*video* safety dataset, so these are in-distribution for T2V unlike I2P's art prompts.
  Subset selected by **our** documented keyword filter over SafeSora's `safety_critical` prompts —
  SafeSora publishes no per-harm-category labels, so this must not be described as "SafeSora's
  nudity category". Seeds assigned by us (SHA-256 of prompt text) and committed.

`control_unrelated_prompts` is the same fire-era unrelated set exp063 used, so the collateral
column stays comparable to our existing runs. No `related` set — the only nudity-adjacent prompts
we have are exp079's *training* retention anchors, and scoring on those once they're trained
against would be evaluating on the training set (open gap, see the doc).

## What to watch
- **Does the base model actually produce nudity on these prompts?** This is the gate, exactly like
  exp064 was for the ImageNet pilot. If base-model detection is low on I2P, the erasure numbers on
  it are meaningless (nothing to erase) — most plausible failure mode is that I2P's image-style art
  prompts ("art by ...") don't drive a T2V model the way they drive SD-1.4. SafeSora, being
  video-native and blunt, is the safety net if that happens.
- **How the two sets compare to each other and to our in-house set.** A large gap between
  in-house and external base rates is itself a finding worth reporting — it would say our own
  prompt set is unusually easy to elicit the concept from.
- NudeNet's known unreliability applies here as much as anywhere
  ([[feedback-detector-metrics-not-ground-truth]]) — but for the *base* row it matters less, since
  we report it as the reference the same detector produces for every other row.

## Status
- [x] Prompt sets built and committed (`tools/build_external_nudity_evalsets.py`, deterministic).
- [x] Config drafted; grid + `Config` construction verified end-to-end locally.
- [x] Submitted; all grid runs generated every clip, then timed out during scoring (see above).
- [x] Scored post-hoc with `tools/score_eval_videos.py` — no regeneration needed.
- [x] Base rates recorded per set; both pass the gate (see Results).
- [x] Paired with exp083 (NegPrompt).
- [ ] Paired with our own checkpoint, once exp080 picks an LR.
- [ ] DOVER filled in post-hoc (`tools/score_dover.py` on x86_64) — scored on helios, so 0.0.
- [ ] Visual spot-check of a sample of flagged clips ([[feedback-detector-metrics-not-ground-truth]]).

## Results (2026-08-07) — both benchmarks pass the gate

| set | n | nudity rate | clip | colorfulness | motion |
|---|---|---|---|---|---|
| I2P (`run_001`) | 95 | **0.326** | 0.2596 | 38.07 | 0.779 |
| SafeSora (`run_002`) | 100 | **0.480** | 0.2752 | 37.54 | 1.662 |
| unrelated (both runs) | 15 | 0.000 | 0.3309 | 33.76 | 2.013 |

**The gate passes on both, SafeSora more strongly.** The risk this experiment existed to test was that
I2P's image-era art prompts would not drive a T2V model the way they drive SD-1.4, leaving nothing to
erase. 0.326 is plenty. SafeSora's 0.480 is higher in the predicted direction — it is video-native
and blunt where I2P is stylised and art-historical.

**This is the first nudity eval with usable statistical power.** Our in-house set runs at n=10, where
one clip is 10 percentage points; exp073's trajectory over five checkpoints (0.000, 0.100, 0.100,
0.100, 0.300) is consistent with pure noise. At n=95 the standard error is +/-4.8pp rather than
+/-14.5pp. Erasure claims should be made on these sets, not on `cogvideox_nudity.csv`.

**The unrelated row is a specificity check, and NudeNet passes it: 0.000 over 15 clips.** Read
together with exp079 (0.844 on a red bikini, 12 of 13 flags false on near-miss content) this locates
the detector's failure precisely: it is *not* indiscriminate — it is silent on ordinary content and
over-fires specifically in the near-miss band (swimwear, sports bras, tight crops). That is a
sharper and more defensible claim than "NudeNet is noisy," and it is the argument for reporting a
`related` column.

**Determinism confirmed for free.** The `unrelated` row is identical across `run_001` and `run_002`
— 0.000 / 0.3309 / 33.76 / 2.013 from two independent jobs on the same prompts and seeds. The
per-prompt seed policy holds end to end.

**Caveat worth a sentence in the paper: I2P's motion is 0.779, well under half the unrelated set's
2.013.** Those art prompts produce near-static clips, so I2P is partly out of distribution for a
video model and erasure measured there says less about motion-heavy content. This is an argument for
reporting both benchmarks rather than choosing one; SafeSora at 1.662 covers the gap.

## Run 1 (2026-08-07): timed out in scoring — videos intact, no regeneration needed

All grid runs hit `CANCELLED ... DUE TO TIME LIMIT` at the 6h budget. An eval job generates on GPU
first and scores on CPU last, so a too-small budget kills it *after* the expensive part. Every clip
was written before the kill — 95/95 (I2P), 100/100 (SafeSora), 15/15 unrelated in each run, verified
on the cluster — and only `metrics.json` is missing.

**Two causes, not one.** The 6h `slurm_time` was too small (my error). But the scoring tail was also
far slower than it should have been: NudeNet built its ONNX session with no `SessionOptions`, so
onnxruntime sized the thread pool to helios' 288 cores for a 320x320 nano model, and dispatch
overhead swamped the actual inference. Fixed in `zml/benchmarks/check_for_nudity.py` — **8.7x
faster, bit-identical scores** (see that commit for the measurements). Every helios eval run before
that commit paid this cost, so slow CPU tails in earlier runs should not be read as "scoring is
inherently expensive."

So the ~24 h of GPU work is intact and does **not** need repeating. Recover with
`tools/score_eval_videos.py` (added for this), which reads the run's own config to pair each clip
with the prompt that generated it and writes the same `metrics.json` via the same `score_video_dir`
the eval path itself uses — a recovered run is not scored differently from a normal one:

```
uv run python tools/score_eval_videos.py <grid_dir>/run_001 <grid_dir>/run_002
```

It needs the videos, which are still cluster-side (the last pull excluded them). Run it on the
cluster, or pull those `eval_step_0/` dirs first. DOVER only contributes on x86_64, so if scored on
helios its fields stay 0.0 and can be filled later with `tools/score_dover.py`.

`slurm_time` raised 6h → 12h so a resubmission does not repeat this.
