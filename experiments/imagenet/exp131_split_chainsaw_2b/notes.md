---
status: ready
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  Not yet run.
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

## Downstream
A passing build is the dataset for a 2B chain-saw `frame_replace` training run (exp069's role, ported
to 2B). A collapse in yield is `needs_human` territory only if it looks like a genuine 2B limitation
rather than a knob this thread already has a fix for (trajectory mode, prompt reframing, more seeds).

## Status
- [ ] Submitted.
- [ ] Screened (`tools/screen_split_dataset.py`); pass rate compared against exp127's 83%.
- [ ] Region balance checked.
