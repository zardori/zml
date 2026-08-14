---
status: done
concept: face
method: frame_replace_split/precompute
thread: face_identity
takeaway: >
  Scale-up of exp115's 9-triple Obama dataset toward exp095's ~30-target need. exp115's own
  metadata shows the 30% yield is a framing problem, not a split_step_frac problem, so this keeps
  split_step_frac at 0.8 and tests the hypothesis with 3 CSVs: a re-seed of the existing 30
  prompts (isolates seed variance) plus 60 new medium/close frontal-framed prompts. DONE: confirmed
  -- reseed reproduces exp115's exact 30% baseline (9/30, same prompts, different seeds), while the
  two framing-controlled CSVs nearly double it (15/30 = 50%, 19/30 = 63%). 43/90 kept here, 52
  total combined with exp115's 9 -- comfortably above the ~30 target. Filtered sets at
  `metadata_human_filtered_run00{1,2,3}.json`. exp095 is repointed and otherwise ready; only
  running `./merge_dataset.sh` on helios to build `combined_dataset/` is left (latents never left
  the cluster).
---
# exp116 — scale-up of the Obama split-prompt frame_replace dataset

## Why
`exp115` (`split_step_frac: 0.8`) fixed exp092's chimera-face problem but kept only 9 of 30 triples
on human review -- too thin for `exp095_frame_replace_obama`'s training run, which wants a dataset
comparable in size to the nudity build its regime was copied from field-for-field.

Before reaching for another `split_step_frac` sweep, cross-referencing exp115's own
`outputs_20260812_141127/metadata.json` against its keep list (seeds 7406, 7407, 7410, 7413, 7417,
7421, 7425, 7427, 7428) shows the dominant failure is upstream of the sampler entirely:

| screen | 9 keeps | 21 rejects |
|---|---|---|
| `original_max_confidence >= 0.30` AND `variants.wholeclip` A-side max confidence `>= 0.30` | 9/9 pass | 14/21 cut |

14 of the 21 rejects have `original_max_confidence` at or near **0.000** -- ArcFace finds no
recognizable Obama in the split clip's concept region at all. These are wide, side-on or occluded
framings (golf swing, gallery back-view, boarding stairs turned away, recording booth, tarmac). All
9 keeps are medium/close, frontal, person-centred shots with `original_max_confidence >= 0.344`.

So the loss is **the base model not rendering a recognizable Obama for that (prompt, seed)**,
nothing to do with `split_step_frac`. This run keeps `0.8` and instead controls for framing.

## Setup
Three new CSVs (`prompt_a,prompt_b,prompt_c,seed` schema, `split_face_*` prefix so
`tools/split_face_prompts.py`'s anti-eval-leak check covers them automatically):

- `prompts/face_identities/split/barack_obama_reseed.csv` -- the existing 30 A/B/C triples verbatim, seeds
  7801-7830. Zero authoring; isolates seed variance from prompt variance -- if this CSV's yield is
  also ~30%, the framing hypothesis needs revisiting.
- `prompts/face_identities/split/barack_obama_closeup1.csv` / `_closeup2.csv` -- 60 new prompts, seeds
  7831-7890, all medium/close and frontal (subject facing or near-facing the camera, no back-views,
  no profile-only actions, no occlusion), with the B-substitute varying per row (age/hair/build/
  clothing) and C keeping an unnamed person (never dropping the person -- the face-specific
  deviation from the nudity recipe, `docs/face_identity.md` §4.4).

Config is `exp115`'s field-for-field (`split_latent_frame: 7`, `concept_region: random`,
`split_jitter: 2`, `split_step_frac: 0.8`, `emit_whole_clip_target: true`) with only `csv_path`
changed to the list above, so `submit_job.py` grids it into 3 parallel helios jobs.

Seeds 7801-7890 are disjoint from every other `prompts/*.csv` (Obama 7401-7430, Merkel 7501-7530,
preservation 7601-7625, Elizabeth 7701-7730, eval 1065-5593) -- checked programmatically before
committing these CSVs.

## Automated pre-screen
`tools/screen_split_face_dataset.py` reads `original_max_confidence` and the wholeclip A-side max
confidence straight out of `metadata.json` (both already written by
`frame_replace_split_precompute.py`, no GPU needed) and flags which rows are worth a human's time.
Calibrated against exp115: `--metadata
experiments/face_identity/exp115_split_face_obama_dataset_frac08/outputs_20260812_141127/metadata.json` passes
16/30, including **all 9** human keeps and **zero** false rejections -- it only cuts the "no face
rendered at all" cases, not the borderline ones a human should still judge. It is a triage, not a
verdict: `docs/split_prompt.md` documents that gating on the detector *inside* precompute cost
exp078 half its yield, so this only decides what to watch, not what to keep.

## What to watch
Same as exp115/exp092 -- splice quality (`*_original.mp4` vs `*_edited.mp4`) and whole-clip identity
separation (`*_wholeclip_a.mp4` vs `*_wholeclip_b.mp4`) reviewed separately, but only for the rows
the screen tool flags as survivors. Also worth tracking per-CSV yield separately (reseed vs.
closeup1 vs. closeup2) to see whether the framing rewrite actually moved the needle over a same-
prompt re-seed.

## Results (2026-08-14) — 43/90 kept (48%), reseed reproduces exp115, framing roughly doubles yield

Submitted as a 3-way grid (`grid_20260813_173003`), one job per CSV, 30/30/30 rows, 0 skips in any
run. Reviewed all 90 triples clip by clip (splice quality + whole-clip identity separation, same
protocol as exp115).

| run | CSV | kept | yield |
|---|---|---|---|
| run_001 | `split_face_barack_obama_reseed.csv` (existing 30 prompts, new seeds) | 9/30 | 30% |
| run_002 | `split_face_barack_obama_closeup1.csv` (new, framing-controlled) | 15/30 | 50% |
| run_003 | `split_face_barack_obama_closeup2.csv` (new, framing-controlled) | 19/30 | 63% |

Kept seeds:
- run_001: 7801, 7806, 7807, 7809, 7810, 7813, 7817, 7820, 7825
- run_002: 7831, 7832, 7833, 7834, 7836, 7837, 7838, 7839, 7840, 7843, 7848, 7850, 7854, 7858, 7860
- run_003: 7861, 7863, 7864, 7866, 7867, 7868, 7869, 7870, 7871, 7872, 7875, 7876, 7877, 7879, 7880,
  7881, 7882, 7884, 7886

Filtered metadata at `metadata_human_filtered_run00{1,2,3}.json` (experiment root, git-tracked),
written with `tools/filter_retention_metadata.py --allow-skew` (same override reason as exp115 and
exp109: the 50% guard is calibrated for retention sets, not split-prompt triples).

**The reseed-vs-framing comparison confirms the framing hypothesis directly**: run_001 (the exact
same 30 prompts as exp115/exp092, only the seeds differ) lands on **exactly** exp115's 30% baseline
-- seed variance alone buys nothing. The two framing-controlled CSVs (medium/close, frontal,
no-occlusion) land at 50% and 63%, 1.7-2.1x the baseline. Writing for framing is a real, repeatable
lever on yield for this concept, not noise.

**Screen-tool calibration note**: `tools/screen_split_face_dataset.py`'s exp115-tuned thresholds
(0.30/0.30) had zero false rejections on exp115 but produced 6 false rejections here out of 43 human
keeps (~14%) -- 1/9 in run_001, 2/15 in run_002, 3/19 in run_003 -- all in triples that scored just
under 0.30 but still passed human review. Consistent with its own docstring (a triage, not a
verdict): useful for cutting obvious no-face rejects, but not a substitute for watching borderline
clips. Not retuning the threshold on this run's data to avoid fitting it to two datasets at once;
revisit if a third Obama batch or the Elizabeth build (exp093) shows the same pattern.

## Downstream
1. ~~Filter each of the 3 runs~~ — done above.
2. Merge all 4 filtered sets (exp115's 9 + this run's 43) into `combined_dataset/` with the new
   local entrypoint `./merge_dataset.sh` (repo root), which ssh's to helios and runs
   `zml/precompute/merge_frame_replace_datasets.py` there — the `.pt` latents never left the
   cluster (`pull_results.sh` excludes them by default), so the merge has to happen where they
   live. Exact command is in `exp095_frame_replace_obama/config.yaml`'s header comment.
   `merge_frame_replace_datasets.py` now resolves sources through `zml.paths.resolve_input_path`
   (peer-root fallback) and rejects a missing source instead of writing a dangling symlink.
3. ~~Repoint `exp095_frame_replace_obama/config.yaml`'s `metadata_file`/`latents_dir` at
   `combined_dataset/`~~ — done; `retention_metadata_file`/`retention_latents_dir` also repointed
   at exp094's real completed output dir. Only running the merge itself is left.
4. `docs/face_identity.md` §5's `split_step_frac` bullet updated to reflect exp115/exp116.

## Status
- [x] Submitted.
- [x] Dataset reviewed per-run — screen-tool survivors watched, splice + whole-clip quality (see
      Results above).
- [x] Per-CSV yield compared (reseed 30% vs. closeup1 50% vs. closeup2 63%) — framing hypothesis
      confirmed.
- [ ] Merged with exp115 into `combined_dataset/` via `./merge_dataset.sh` (needs the cluster login
      node, one command — see `exp095/config.yaml`'s header). exp095 is already repointed at the
      resulting path and otherwise ready to submit once this runs.
