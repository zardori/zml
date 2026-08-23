---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  BLOCKED ON exp145 AND ITS GATE. Trains exp110's exact regime (eta 2.0, rank 8, lr 1e-4, fire
  retention) on the gen5 uniform-wardrobe dataset instead of gen4's varied one — donor-wardrobe
  diversity is the only variable. Do not submit unless exp145's measured edit coherence lands
  clearly above OLD-31's 0.714; below that the manipulation failed at the data level.
---
# exp146 — frame_replace on gen5 (uniform wardrobe)

## Why
The training arm of exp145. See that experiment's notes for why a uniform wardrobe is worth testing
even though the LAB edit statistics came back negative: it is a deliberate manipulation of the one
variable with a mechanistic story, not a finding being followed up.

## What is held fixed
Everything except the dataset. `config.yaml` copies exp110's training fields verbatim — eta 2.0,
rank 8, alpha 8, lr 1e-4 constant, 200 steps, `save_interval` 10, grad-accum 4, velocity loss on
original latents, timesteps 400-1000, `retention_weight` 1.0 against exp041's fire anchors, seed 42.
The comparison is exp146 vs exp110, same regime, different donor diversity.

Two deliberate departures, neither touching training:
- **`eval_num_prompts: 25`, not exp110's 10.** Ten videos cannot resolve a detection rate; exp110's
  two "0.0000" checkpoints were an artifact of it and were reported as a new best before the full
  sets contradicted them.
- **`control_related_prompts` / `control_unrelated_prompts` added.** A uniform donor set is the
  configuration most likely to teach "render this navy shirt" rather than "do not render nudity",
  and exp113 measured the gen4 checkpoint shifting colour on nudity-free prompts (53.1 vs base
  45.8). Without these sets that collateral is invisible until a separate eval job.

## What counts as success
Not a lower nudity rate on its own. gen4-derived runs already erase *deeper* than the incumbent
(exp123 r1 s80 = 0.070, exp136 r1 s200 = 0.040, vs exp080 r2 s120 = 0.120) — they just only do it
inside the degeneracy trough and hand it back as sharpness returns. The thing exp080 has and they do
not is a checkpoint that is erased **and** sharp simultaneously.

So the read is: **is there a checkpoint with a low `nudity_frame_rate` at DOVER-t >= 0.058?** For
reference, at each run's first checkpoint clearing that bar the frame rate is 0.000 for exp080 s120
and exp110 s120-140, but 0.23 for exp123 r1 and exp136 r1 and 0.32 for exp123 r2. DOVER reads 0.0 on
helios (aarch64) — score post-hoc with `tools/score_dover.py`, and never read colorfulness as
quality.

## Status
Not submitted; blocked on exp145 producing and passing its gate.
