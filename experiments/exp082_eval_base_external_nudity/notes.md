---
status: ready
concept: nudity
method: eval
thread: nudity
takeaway: >
  Base-model ("Original") reference on two EXTERNAL nudity benchmarks — I2P (95 prompts, real
  benchmark, its own seeds) and SafeSora (100 prompts, video-native). Needed because every prior
  nudity number is on prompts we wrote ourselves: cogvideox_nudity.csv shares ZERO prompts with
  real I2P despite being called "i2p-derived". Not yet submitted.
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
- [ ] Scored — run tools/score_eval_videos.py against the existing videos (no regeneration needed).
- [ ] Base rates recorded per set; compared against our in-house `cogvideox_nudity.csv` rate.
- [ ] Paired with exp083 (NegPrompt) and, once exp080 picks an LR, our own checkpoint.

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
