# exp059 — split-prompt generation for partial-nudity clips (milestone 1: generate & inspect)

## Why
frame_replace needs *partial*-concept clips (concept in some frames + concept-free donor frames in
the same clip). Fire gives this naturally (it flickers); nudity does not — a naked body is present the
whole clip, so there are no donors and the method can't build a target. This is why frame_replace
hasn't transferred to nudity, and why the single split-prompt CSVs (partial_nudity*.csv) were
unreliable — one prompt asking for a clothed→naked transition, which the model usually ignores.

New precompute algorithm (`zml/precompute/split_prompt_precompute.py`, method `split_prompt`):
**manufacture** the partiality. Steer the first temporal half with a concept-free prompt (B) and the
second half with a concept prompt (A) during the early (content-setting) denoising steps, then heal
the temporal seam by conditioning the tail of the schedule on a shared neutral prompt (C). Done
MultiDiffusion-style — two transformer forwards per early step, spliced per latent-frame region, one
scheduler step on the full latent (keeps the DPM-solver state coherent). No attention surgery.

## This experiment (milestone 1)
Generate and save **4 clips per row** — A (naked), B (clothed), C (neutral), and the combined split
clip — from `prompts/split_nudity.csv` (4 minimal A/B/C triples: same scene + seed, differ only in
clothing). All four share one initial noise per row, so they are directly comparable (and A/B double
as a paired same-seed donor baseline). No detection / donor-edit yet — this milestone only answers:
**does the splice produce a coherent clip that is clothed early and naked late?**

Knobs: `split_latent_frame` (7 → frames [:7] clothed, [7:] naked) and `split_step_frac` (0.5 → split
for the first half of steps, then prompt C). Both are the likely-decisive hyperparameters.

## What to look at
- `videos/pN_sM_combined.mp4` vs `_A/_B/_C.mp4`. Want: combined = clothed first half, naked second
  half, one coherent person/pose/scene, seam healed.
- Failure signs: both halves collapse to one state (split ended too early / C washed the concept out);
  visible hard seam (C phase too short); concept leaks into the clothed half.
- If it works → milestone 2: run a nudity detector, donor-edit the naked frames, save as a
  frame_replace dataset (mirrors `frame_replace_precompute.py`). If not → sweep split_step_frac /
  split_latent_frame, or fall back to the paired same-seed A/B baseline (A with B as frame-wise donor).

## Status
- [ ] Submitted.
- [ ] Videos inspected (does the splice work? best split_step_frac / split_latent_frame).
