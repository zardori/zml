---
status: ready
concept: face
method: frame_replace_split/precompute
thread: face_identity
takeaway: >
  Rebuild of exp093's unusable Queen Elizabeth II dataset (by eye 1/30 correct, screen keeps 0/30).
  Not yet submitted. exp093 had three independent, already-fixed defects: split_step_frac 0.5 (the
  chimera-face value exp115 rejected), pre-exp116 wide/ceremonial prompt framing, and an
  Obama-calibrated absolute screen. This run fixes all three (split_step_frac 0.8,
  framing-controlled CSVs written against her own eval prompts, tools/screen_split_dataset.py's
  identity-scale-free contrast index) and additionally grids split_mode {prediction, trajectory} --
  exp127/exp131's yield fix, never tried on faces -- against the same 3 CSVs/seeds. See Gates below.
---
# exp180 — rebuild of the Queen Elizabeth II split-prompt frame_replace dataset

## Why
exp093 built 30/30 with zero skips but screens at 0/30, and human review confirms only `p0_s7701`
has a correct split. exp093's own notes framed this as a threshold question ("the gate is calibrated
on Obama") and left an open fork (rescale the gate per identity, or swap the pilot identity).
Re-reading it against everything the thread has learned since (exp115/exp116 for Obama, exp120/
exp127/exp131 for objects) shows the diagnosis was incomplete — there are three independent, already-
understood defects, not one threshold mismatch:

1. **Sampler.** `exp093/config.yaml` sets `split_step_frac: 0.5`. exp092's human review found 0.5
   under-heals and produces a chimera face blending the identity and the B-prompt substitute; exp115
   fixed this at `0.8` for Obama. exp093's config comment even says it should "keep in sync with
   exp092's final decision" — it never did. `docs/face_identity.md` §5 already names this:
   *"Elizabeth (exp093) is still configured at 0.5 and should move to 0.8 ... before submitting."*
2. **Prompt framing.** `prompts/face_identities/split/queen_elizabeth_ii.csv` is written in the
   pre-exp116 style: wide, ceremonial, side-on, crowd-distance scenes ("From a raised reviewing
   stand", "Riding in an open state coach", "marches past in formation"). Nine of its 30 rows read
   `original_max_confidence` exactly 0.000. exp116 measured this exact failure class capping Obama's
   yield at 30%, and fixed it by rewriting for medium/close frontal framing (30% → 50%/63%). Her
   published eval set (`prompts/face_cogvideox.csv`, 30 rows) is itself overwhelmingly seated,
   indoor, medium-close on the face — the training prompts should look like it and did not
   (`docs/split_prompt.md` §6: "write the CSV against the eval prompts, not from scratch").
3. **Screen.** exp093 used `tools/screen_split_face_dataset.py`'s absolute 0.30 gate, calibrated on
   Obama's base id_sim (0.5082); hers is 0.3272, so a gate tuned on him sits above her own ceiling.
   `docs/split_prompt.md` §3.1 now documents that tool as the superseded, absolute-threshold-only
   ancestor and says new work should use `tools/screen_split_dataset.py`'s within-clip contrast
   index instead, which compares each clip's A-half against its own B-half and is therefore
   identity-scale-free by construction — this answers exp093's fork #1 without inventing a
   per-identity threshold table.

A fourth, cheaper defect found while rewriting: exp093's B-prompt substitutes are themselves
Elizabeth-shaped ("an elderly woman in a formal gown and diamond tiara", "a slender, silver-haired
woman in a tweed jacket", "a composed, elderly woman in a heavy wool coat and fur hat"). This is
exactly exp067's church failure (substitute buildings were church-shaped, driving `not-split`
verdicts); the new CSVs give B equally detailed but non-identifying descriptions (no tiara, no
pearls, no royal-coat silhouette).

**New lever, untested on faces:** `split_mode: trajectory` (exp127) took exp120's 12 known-suppressed
object rows from 0/12 to 12/12 at identical compute (two transformer calls per split step either
way), and exp131 confirmed it transfers across base models (5b → 2B). exp093's own closing line —
*"the whole-clip A gate does clear 0.30 on 6 rows... it is the split that loses her"* — is a textbook
description of the shared-latent suppression pathology trajectory mode was built to cure. Rather than
assume it wins, this run grids `split_mode ∈ {prediction, trajectory}` against the same 3 CSVs and
seeds, so the sampler comparison is clean and unconfounded with prompt content.

No re-seed control arm (unlike exp116). A prompt-vs-sampler attribution against exp093 is already
confounded by exp093's different `split_step_frac`, so that comparison is not attempted here; the
size and quality of the resulting dataset are what matters for unblocking exp096/exp098.

## Setup
Three new CSVs, `prompt_a,prompt_b,prompt_c,seed` schema, at
`prompts/face_identities/split/queen_elizabeth_ii_closeup{1,2,3}.csv` — 30 hand-authored triples
each, seeds 7901-7930 / 7931-7960 / 7961-7990 (disjoint from every other `prompts/*.csv`: Obama
7401-7430/7801-7890, Merkel 7501-7530, preservation 7601-7625, Elizabeth 7701-7730, eval 1065-5593 —
checked programmatically). Written against her 30 published eval prompts' register (seated, indoors,
ornate/dimly-lit rooms, at a desk or in an armchair), copying exp116's `barack_obama_closeup2.csv`
template (its 63%-yield arm): a shot-size phrase first ("In a medium close-up...", "Close to
camera...", "In a tight medium shot..."), then a small domestic/ceremonial action, ending on an
explicit static-camera sentence. A keeps her identifying accessories (hat, pearls, brooch, tiara,
gloves) varying per row, matching how the eval prompts name them; B is an anonymous woman with
different age/hair/build/clothing and none of those accessories; C keeps an unnamed person in every
row (the face-axis deviation from nudity/objects, `docs/face_identity.md` §4.4 — the heal phase
conditions the whole latent on C, so a person-free C would push the face out of the concept half
too). Anti-cheat checked: `uv run python tools/split_face_prompts.py` → "Anti-cheat check passed."

Config is exp116's field-for-field (`split_latent_frame: 7`, `concept_region: random`,
`split_jitter: 2`, `emit_whole_clip_target: true`, `tail_prompt_mode: c`, `concept_guidance_scale`
unset) with `split_step_frac: 0.8` and a 2-way `split_mode` grid layered on top of the 3-way
`csv_path` grid — `submit_job.py`'s `expand_grid` Cartesian-products every list field, so this is 6
jobs with no new machinery. `expand_grid` iterates the last-listed key fastest, so runs should land
as `(csv1,pred) (csv1,traj) (csv2,pred) (csv2,traj) (csv3,pred) (csv3,traj)` — confirm against each
`run_00N/config.yaml` after submission rather than assuming the order.

## Pre-registered gates
- **G1 — does trajectory transfer to faces?** Pooled over the 90 rows per arm, `trajectory` pass
  rate ≥ `prediction` pass rate + 15 points ⇒ adopt trajectory for the face axis and record it in
  `docs/split_prompt.md` §3.3.1 (exp127 moved 33% → 94% on objects; 15pp is a deliberately modest
  bar for a first face measurement).
- **G2 — did the reframing fix rendering at all?** Fraction of rows with whole-clip A-side max
  confidence ≥ 0.30 (exp093's own gate, applied here to the diagnostic, not the keep decision).
  exp093 got 6/30 (20%). ≥50% ⇒ the prompts were the bottleneck and are now fixed. Still ~20% ⇒ the
  ceiling is genuinely the base model's grasp of this identity, and exp093's fork #2 (swap the
  second pilot identity to Trump, base id_sim 0.4875, or Biden, 0.4504) is the honest call to make
  next — report that plainly rather than lowering any gate to manufacture a pass.
- **G3 — size.** ≥30 human-reviewed keeps total across the winning arm(s) (exp095 trained on 52).
- **G4 — balance.** Surviving `concept_region` roughly balanced first/second (printed by
  `screen_split_dataset.py`); a skewed keep set teaches the positional shortcut
  (`docs/split_prompt.md` §4).

## What to watch
Splice quality (`*_original.mp4` vs `*_edited.mp4`) and whole-clip identity separation
(`*_wholeclip_a.mp4` vs `*_wholeclip_b.mp4`) reviewed separately, same protocol as exp115/exp116.
Also: whether `trajectory` mode introduces a visibly different seam-coherence failure — its own
docstring flags this as the open question (the seam currently gets its coherence from the shared
noise *and* the shared latent; trajectory removes the latter).

**Caveat for the screen:** `VideoFaceDetector.frame_confidences` returns `0.0` for a *no-face*
frame, not "detected and not her" (`docs/face_identity.md` §3.1's convention). So on this concept
`screen_split_dataset.py`'s contrast index is partly a face-*presence* differential, not purely an
identity one — worth stating explicitly since the tool was calibrated on object detectors that don't
have this convention. Primary screen: `--min-concept-max 0.23` (`IDENTITY_THRESHOLD` from
`zml/benchmarks/check_for_face.py`, FPR 0%/TPR 78% per exp090's calibration) and default
`--min-contrast-index 0.4`; also run the default `--min-concept-max 0.10` as a sensitivity check, and
`tools/screen_split_face_dataset.py` for continuity with exp115/exp116's published numbers.

## Downstream
Feeds `exp096_frame_replace_elizabeth` — once merged, repoint its `metadata_file`/`latents_dir` at
this run's `combined_dataset/` and set `target_variant: split` (exp095's confirmed winner for
Obama; exp096's config currently carries this as an explicit placeholder). `exp098` follows exp096.
If the two `split_mode` arms are not merged together (see Gates — mixing sampler regimes in one
training set is avoided if one arm clearly wins), record which single arm feeds `combined_dataset/`
and why.

## Status
- [x] Three framing-controlled CSVs authored (90 triples total, seeds 7901-7990) and anti-cheat
      checked.
- [x] `experiments/face_identity/exp180_split_face_elizabeth_scaleup/config.yaml` written (6-way
      grid: 3 CSVs × 2 split_modes, `split_step_frac: 0.8`).
- [ ] Submitted (project owners submit jobs, not Claude — see `.claude/rules/workflow_rules.md`).
- [ ] Built and pulled.
- [ ] Screened with both `screen_split_dataset.py` (primary) and `screen_split_face_dataset.py`
      (continuity check).
- [ ] Survivors watched by eye, keeps written with `filter_retention_metadata.py --allow-skew`.
- [ ] Gates G1-G4 evaluated and written up here.
- [ ] Merged via `./merge_dataset.sh`; `exp096`/`exp098` repointed.
- [ ] `exp093`'s notes.md marked superseded; `docs/face_identity.md` §5/§6 and `docs/split_prompt.md`
      §3.1/§3.3.1 updated; `experiments/INDEX.md` regenerated.
