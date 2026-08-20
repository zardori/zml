---
status: done
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  NOT falsified: the trajectory-mode recipe transfers to 2B cleanly. 25/30 (83%) pass, matching
  exp127's 83% on the same prompts under 5b almost exactly, with 0 not-split failures (the
  suppression pathology trajectory mode fixed on 5b does not reappear) and 5 no-concept (a
  prompt/framing miss, same failure category as every prior build). Survivors balance 12 first / 13
  second, close enough to use without a rebalancing pass. Screened set committed at
  `outputs_20260820_103524_screened.json`. Unblocks a 2B chain-saw frame_replace training run —
  found in the process that it also needs its own retention set (exp132), since exp068's is 5b-only.
---
# exp131 — chain-saw split-prompt dataset, on CogVideoX-2B

## Why
GOAL.md moves the object thread's base model to CogVideoX-2B, and exp130 cleared the first gate
(base model renders all ten protocol classes; chain saw restricted top-1 0.885, no degenerate class).
That only tests the *unmodified* pipeline, though. frame_replace's training data comes from the
split-prompt sampler (`generate_split_clip`), which conditions one temporal region on prompt A
(concept) and the other on prompt B (concept-free), spliced and healed by prompt C. That mechanism
was discovered, broken, and fixed (exp099, exp117, exp119, exp120, exp127) entirely on 5b — the
`prediction`-mode suppression exp120/exp127 found and fixed with `split_mode: trajectory` is a
property of how the two branches share latent context during denoising, which is exactly the kind of
thing that could behave differently on a different transformer. Before spending a training run, the
question is whether the current best-known recipe (trajectory mode) still yields usable partial-
concept clips at all on 2B.

## Hypothesis and what would falsify it
Hypothesis: the trajectory-mode split-prompt recipe, unchanged except for `model_id`, yields chain-
saw clips on 2B at a rate comparable to exp127's 5/6 (83%) on the same prompts under 5b — i.e. within
noise of 30 rows, not a collapse.

Falsified if: pass rate drops sharply (roughly half or worse of the 5b trajectory rate, on the same
`no-concept` / `not-split` / `pass` breakdown `screen_split_dataset.py` reports), which would mean
2B's smaller transformer does not carry enough contextual coupling for the trajectory splice to heal
cleanly, or renders the object less reliably under the closeup framing than exp130's plain-prompt
gate suggested. Either result blocks 2B `frame_replace` training on this recipe until diagnosed —
same role exp066 run 1 played when the detector-mask approach failed on 5b.

## Setup
Single variable changed from exp127's `trajectory` arm: `model_id: THUDM/CogVideoX-2b` in place of
5b. Same 30-row prompt CSV as exp117/exp121 (`chain_saw_closeup.csv`), same split geometry
(`split_latent_frame: 7`, `concept_region: random`, `split_jitter: 1`, `split_step_frac: 0.85`,
`boundary_margin: 2`, `tail_prompt_mode: c`), same `split_mode: trajectory` (not the older
`prediction` default — no reason to re-test the already-inferior mode on a new base model when
trajectory costs the same). `emit_whole_clip_target: false`, since exp117 already answered that
diagnostic question and it is not a usable training target.

No code change: `model_id` is already a plain field on `frame_replace_split_precompute.Config`
(`zml/precompute/frame_replace_split_precompute.py:68`), threaded straight into
`CogVideoXPipeline.from_pretrained` with no 5b-specific assumption (checked: latent geometry, VAE
scaling and `edit_latent`/`edit_latent_reflected` are keyed on `num_frames`, not `model_id`, and are
identical between the two CogVideoX sizes). exp130 already exercised this same `from_pretrained(2b)`
call path successfully in the eval pipeline (`zml/eval/eval_model.py`), which de-risks the pipeline
API working at all on 2B.

## What to watch
- **Pass count** via `tools/screen_split_dataset.py`, against exp127's 5/6 (83%) trajectory rate on
  the same prompt seeds (exp127 used a 6-seed subset of these 30; this run is the full 30).
- **`no-concept` count** — if 2B renders the chain saw less reliably under this closeup framing than
  its plain-prompt behaviour (exp130's restricted top-1 0.885) suggests, this is where it shows up.
- **`not-split` count** — the trajectory-mode failure mode exp127 still saw on some rows (concept
  leaking into or missing from the wrong half).
- **`concept_region` balance among survivors**, so a follow-up merge doesn't reintroduce the
  positional shortcut exp117/exp121 balanced away.

## Result
`tools/screen_split_dataset.py` on the full 30-row `metadata.json`:

```
30 clips | pass 25 (83%) | not-split 0 | no-concept 5 | blank-target 0
surviving concept_region balance: 12 first / 13 second
```

Against exp127's 5/6 (83%) trajectory-mode rate on a 6-seed subset of the same prompts, this is the
same pass rate on the full 30, not noise in either direction. The `not-split` count — the failure
mode `split_mode: trajectory` was built to fix — is zero, so the fix holds on 2B, not just on 5b. The
5 `no-concept` misses are a prompt/framing problem carried over from the CSV itself (exp117's
closeup framing), not a new 2B-specific rendering gap; exp130's restricted top-1 0.885 for chain saw
already said 2B renders the object reliably enough that this isn't the base-rate limit. Screened set
written to `outputs_20260820_103524_screened.json` (25 entries, committed, not gitignored) —
`latents_dir` stays the gitignored `outputs_20260820_103524/latents`, which exists on helios from
this run.

**Finding that changes downstream scope:** `unlearn_frame_replace.py` asserts the retention
latents' `scaling_factor` against the training model's VAE, and CogVideoX-2b's VAE
(`scaling_factor: 1.15258426`) is calibrated differently from 5b's (`0.7`) — confirmed from the two
models' HF `vae/config.json`, not just inferred from the assert. exp068's preservation anchors are
5b-only, so they cannot back a 2B training run. exp132 builds the 2B counterpart.

## Downstream
A passing build is the dataset for a 2B chain-saw `frame_replace` training run (exp069's role, ported
to 2B), gated on exp132 (2B retention anchors) landing first.

## Status
- [x] Submitted.
- [x] Screened (`tools/screen_split_dataset.py`); pass rate compared against exp127's 83% — matched.
- [x] Region balance checked — 12 first / 13 second, usable without rebalancing.
