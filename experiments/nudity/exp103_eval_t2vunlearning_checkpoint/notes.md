---
status: ready
concept: nudity
method: eval
thread: nudity
takeaway: >
  BLOCKED ON CODE, not on cluster time. Runs T2VUnlearning's own released nudity eraser through our
  eval, so their method and ours are compared on one instrument instead of two papers' self-reported
  numbers. Needs their Receler-style inject_eraser (rank 128) vendored — it is NOT a plain LoRA — and
  the weights fetched from their Google Drive. Config is written; do not submit until both are done.
---
# exp103 — T2VUnlearning's released checkpoint, on our instrument

## Why this is the one comparison that survives review
[`docs/comparability_t2vunlearning.md`](../../docs/comparability_t2vunlearning.md) §3 establishes
that their reported numbers and ours are **not on the same footing**, for two measured reasons:

1. Their Table 1 is very likely the `unsafe = Q16 OR NudeNet` column their scoring script computes,
   not the NudeNet-only rate their paper defines. On our base clips that is 0.516 vs 0.414 — about
   half the 20-point gap between our Original (41.4) and theirs (61.80).
2. They generate with a **CPU** noise generator (`torch.Generator().manual_seed(seed)`), we generate
   with a **CUDA** one. Same seed, completely different noise, completely different videos. Their
   other settings — 50 steps, guidance 6.0, 49 frames, bf16 — match ours exactly, so nothing else is
   left to explain it.

Consequence: we cannot put their row and our row in one table and claim the difference is method.
Running **their weights** through **our** pipeline removes both confounds at once — same generator,
same prompts, same seeds, same detector, same machine — and yields a single self-consistent table
where the only thing varying is the erasure method. It also answers the question the deadline
actually turns on: is their method better than ours, or did they draw different samples?

## What blocks it
**Their eraser is not a LoRA.** `test_cogvideo.py` loads it as:

```python
inject_eraser(pipe.transformer, eraser_ckpt=torch.load(path, map_location='cpu'), eraser_rank=128)
```

from `receler.erasers.cogvideo_erasers` — Receler-style adapter modules injected into the
transformer at rank 128. Our `zml/eval/eval_model.py:build_eval_pipeline` only knows
`PeftModel.from_pretrained`. Two things are needed:

1. **Vendor their `receler/` package** (or reimplement `inject_eraser`), and add a loader branch to
   `build_eval_pipeline` keyed on a new config field — `eraser_checkpoint` / `eraser_rank` — so a
   Receler-style checkpoint and a PEFT LoRA can both be evaluated by the same path. Keep the branch
   in the pipeline builder, not in the eval loop, so nothing downstream learns about it.
2. **Fetch the weights.** They are a Google Drive folder linked from their README, not a HF repo, so
   this is a manual download plus staging on the cluster (and a repo-relative path that
   `slurm/check_config_paths.sh` can see).

Neither is hard; both are real work, and neither is cluster time. `config.yaml` is written and
correct so that when the loader lands this is a submit, not a design task.

## Setup
Mirrors exp101/exp102 field for field — same two prompt sets (their Gen 100, their Ring-A-Bell 79),
same unrelated control, same paired-safe related set, same `eval_inference_steps`. The only
difference from exp102 is which checkpoint is loaded, which is the entire point.

## What to watch
- **Their number on our instrument vs their number in their paper.** If their checkpoint scores
  ~16 here, their metric and ours agree and the Original gap was generation. If it scores much
  higher, their reported row is on the looser `unsafe` metric and their paper's numbers should be
  read accordingly.
- **Their PSR weakness is our opening.** On objects they trade 24 points of PSR-1 for erasure
  (Table 2: 78.38 → 54.03). If that pattern holds for nudity, the paired `related` column here is
  where a localized edit should beat them — and they report no nudity preservation column at all.
- **Motion.** Run `tools/score_dover.py` and read the motion column on their clips too. If their
  method also collapses motion, our −85% is a shared property of the approach rather than our bug,
  and that changes what the paper has to concede.

## Status
- [ ] `receler` vendored + `build_eval_pipeline` loader branch.
- [ ] Weights downloaded and staged on the cluster.
- [ ] `eraser_checkpoint` path filled in (placeholder in config).
- [ ] Submitted (2 jobs).
- [ ] Scored on x86_64, same machine as every other reported row.
