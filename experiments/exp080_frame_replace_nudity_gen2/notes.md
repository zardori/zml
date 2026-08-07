---
status: active
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  First frame_replace run on the expanded dataset (exp061's 21 + exp078's 13 human-approved
  triples = 34). Baseline is exp077: warmup scrapped, learning_rate grid
  [0.00005, 0.0001, 0.0002, 0.0005], 200 steps. Submitted 2026-08-07, 3/4 jobs running, with
  exp041's fire-era retention set; the nudity-specific replacement moved to exp085 as an
  ablation rather than being swapped in silently. Live eval is n=10 - a monitor, not a result.
---
# exp080 — frame_replace nudity, expanded dataset + LR grid

## Why
Direct instruction after exp078's `split_step_frac` grid review: since that review (see exp078's
notes.md) found pass/fail is driven by seed/prompt, not by `split_step_frac` within 0.8-1.0, the
dataset from that grid's `run_005` (1.0) is as good a source as any of the other four values. 13 of
its 50 triples were human-approved (26%) — small enough alone to risk the same generation collapse
exp062 run 2 hit at 12 triples, so this run combines them with exp061's already-confirmed 21
triples (34 total) rather than training on the 13 alone.

Separately, exp077's LR-warmup test (grid `[0, 30, 60]`, see its notes.md) found no decisive
sharpness win and warmup's effect even reversed by step 100 (higher warmup -> lower final
colorfulness). Scrapping warmup and instead grid-searching `learning_rate` directly tests exp073's
own alternative suggestion for the same softness problem: colorfulness was lowest at the earliest
checkpoints under the full 5e-4 step from iteration 1, so a smaller constant LR might fix the same
issue warmup was meant to, without warmup's added complexity.

## Setup
Baseline is **exp077**, not exp062/exp073 — same regime (rank-8 LoRA, `erase_esd_eta: 2`, velocity-
space loss on original latents, `retention_weight: 1.0`, `lr_scheduler: constant`), but:

1. **No `lr_warmup_steps`** — omitted entirely (defaults to 0), matching exp077's own no-warmup arm.
2. **`learning_rate: [0.00005, 0.0001, 0.0002, 0.0005]`** grid — 0.0005 is every prior nudity run's
   value (the baseline arm here); 0.0002 tests exp073's own suggested "smaller LR" alternative to
   warmup. 0.0001 and 0.00005 added on top (compute cost is the same per job, just two more grid
   jobs) so the sweep can tell "smaller is better" from "0.0002 specifically is the right amount
   smaller" — a single smaller value can't distinguish a trend from a coincidence.
3. **`steps: 200`, `save_interval: 20`** (10 checkpoints per run) — double exp073/077's 100-step
   window, to see whether either LR's erasure/sharpness trajectory changes past the point those
   two experiments stopped looking.

**Dataset**: `experiments/exp080_frame_replace_nudity_gen2/combined_dataset/`, built by the new
`zml/precompute/merge_frame_replace_datasets.py` (pure file I/O, no GPU — symlinks each source's
`.pt` files into one `latents/` dir with a per-source prefix to avoid name collisions, writes one
merged `metadata.json`) from:
- exp061's `metadata_human_filtered.json` (21 triples, `outputs_20260802_223148/latents`).
- exp078's `run_005/outputs/metadata_human_filtered.json` (13 triples approved 2026-08-05 from the
  `split_step_frac: 1.0` grid run, `run_005/outputs/latents`).

**This merge step has not been run yet** — the source latents (exp078's specifically) aren't
pulled locally (`pull_results.sh` excludes `.pt` files by default; need `--include-weights`), and
the merge writes symlinks, so it needs to run wherever the actual files live before this config is
submittable. See the command in `config.yaml`'s header comment. `slurm/check_config_paths.sh` will
catch a missing `combined_dataset/` at submission time regardless.

**Retention set is still exp041's fire-era one** (not exp079's nudity-specific set, which exists
but hasn't been submitted/built yet — see exp079's notes.md). Worth swapping once that lands, but
not blocking this run.

`prompts/split_nudity_gen2_approved.csv` (new, committed): the 13 approved rows filtered from
`prompts/split_nudity_gen2.csv`, kept as a durable, reusable file independent of any specific grid
run's output directory — so if the remaining `split_step_frac` values (0.8/0.85/0.9/0.95) get
reviewed later and change which rows are approved, or once a final `split_step_frac` is picked,
this CSV can be regenerated straightforwardly from the prompt level rather than depending on
`run_005`'s specific already-generated latents.

## What to watch
- Whether early-checkpoint softness (colorfulness, and — more reliably per
  [[feedback-detector-metrics-not-ground-truth]] — actual visual sharpness) decreases monotonically
  as LR shrinks across the 4 values, or bottoms out/reverses somewhere in the range, at matched steps.
- Whether erasure still lands (concept detection near 0) by step 200 at every LR, or the smallest
  values (0.00005, 0.0001) are too gentle to erase within this step budget — plausible given they're
  10x/5x smaller than every prior nudity run's LR.
- Generation collapse risk given the 34-triple dataset is still smaller than exp062 run 3's 21 +
  fully-independent regime — watch `motion_score_mean`/`clip_score_mean` per checkpoint the way
  exp062/exp073's collapse checks did, don't trust `concept_detection_rate` alone.
- Whether the exp078-sourced triples (all approved from the same `split_step_frac=1.0` build) behave
  differently in training than exp061's triples — not directly observable from aggregate metrics,
  but worth a qualitative check if erasure or collateral results look unusual.

## Status
- [x] Config drafted (LR grid, 200 steps, no warmup).
- [x] `prompts/split_nudity_gen2_approved.csv` written (13 rows, durable/committed).
- [x] `zml/precompute/merge_frame_replace_datasets.py` written (reusable, not nudity-specific).
- [x] Merge step run on the cluster; `combined_dataset/` built (34 triples).
- [x] Submitted 2026-08-07 — 3 of 4 grid jobs running.
- [ ] Grid completes; LR chosen (human review, not the n=10 detector rate alone).
- [ ] Chosen checkpoint evaluated on the external benchmarks — that, not this run's live eval, is
      the number that goes next to exp082/exp083.

## Run 1 (2026-08-07): submitted as configured above

Submitted with the config exactly as it stands in this folder — exp041's fire-era retention set,
`save_interval: 20`, `eval_num_prompts: 10`, `slurm_time: 16h`. Two improvements were written after
submission and deliberately **not** applied here, so that this file keeps describing the run that
actually executed:

1. **Retention set.** exp079's 20 human-reviewed nudity-adjacent anchors instead of exp041's fire
   near-misses. Moved to **exp085**, which is otherwise identical — making the pair a clean
   ablation of the retention set, which is more useful than the silent swap would have been.
2. **Eval budget.** `eval_num_prompts` 10 -> 20 paid for by `save_interval` 20 -> 40 (same 300
   clips, half the noise per point). Also in exp085.

**Timeout risk, flagged rather than fixed.** Eval fires at every `save_interval`, so this run does
`200/20 * 3 * 10 = 300` clips plus 200 training steps against a 16h budget. exp077 did 150 clips
plus 100 steps in ~7h of an 8h budget, so this is close to the line. If it dies late, checkpoints
are written at every `save_interval` and survive — unlike exp082/exp083, where the timeout landed
after generation and cost the whole scoring pass. Losing the last checkpoint's eval is recoverable;
`tools/score_eval_videos.py` handles it.

**Reading the live eval.** `eval_num_prompts: 10` on `cogvideox_nudity.csv` is a training monitor,
not a result. exp082 showed n=10 is too weak to distinguish anything — exp073's trajectory across
five checkpoints (0.0, 0.1, 0.1, 0.1, 0.3) is consistent with no effect at all. Use it to detect
collapse and to rank the four LRs coarsely; do not quote it.
- [ ] Analysis once results land.
