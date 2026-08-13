---
status: ready
concept: face
method: frame_replace_split/precompute
thread: face_identity
takeaway: >
  Scale-up of exp115's 9-triple Obama dataset toward exp095's ~30-target need. exp115's own
  metadata shows the 30% yield is a framing problem, not a split_step_frac problem (14/21 rejects
  have original_max_confidence ~0.0 -- no recognizable Obama rendered at all, in wide/side-on/
  occluded framings), so this keeps split_step_frac at 0.8 and instead re-seeds the existing 30
  prompts plus adds 60 new medium/close frontal-framed prompts (3x30 CSVs, seeds 7801-7890). Not
  yet submitted.
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

- `prompts/split_face_barack_obama_reseed.csv` -- the existing 30 A/B/C triples verbatim, seeds
  7801-7830. Zero authoring; isolates seed variance from prompt variance -- if this CSV's yield is
  also ~30%, the framing hypothesis needs revisiting.
- `prompts/split_face_barack_obama_closeup1.csv` / `_closeup2.csv` -- 60 new prompts, seeds
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
experiments/exp115_split_face_obama_dataset_frac08/outputs_20260812_141127/metadata.json` passes
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

## Downstream
1. Filter each of the 3 runs with `tools/filter_retention_metadata.py --allow-skew --output
   metadata_human_filtered_run00N.json` (per-run output path, same reason as exp109: the default
   path collides across grid runs).
2. Merge all 4 filtered sets (exp115's 9 + this run's 3) with
   `zml/precompute/merge_frame_replace_datasets.py` into `combined_dataset/` (prior art: exp080
   merged exp061 + exp078 the same way).
3. Repoint `exp095_frame_replace_obama/config.yaml`'s `metadata_file`/`latents_dir` at
   `combined_dataset/` (currently still holds exp092 `outputs_TIMESTAMP` placeholders).
4. Update `docs/face_identity.md` §5's `split_step_frac` bullet, stale since exp115 (still says
   "starts at 0.5 for both pilot identities").

## Status
- [ ] Submitted.
- [ ] Dataset reviewed per-run — screen-tool survivors watched, splice + whole-clip quality.
- [ ] Per-CSV yield compared (reseed vs. closeup1 vs. closeup2).
- [ ] Merged with exp115 into `combined_dataset/`; exp095 repointed.
