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
- [ ] Submitted — manual, per project convention.
- [ ] Base rates recorded per set; compared against our in-house `cogvideox_nudity.csv` rate.
- [ ] Paired with exp083 (NegPrompt) and, once exp080 picks an LR, our own checkpoint.
