---
status: ready
concept: nudity
method: frame_replace_split/reedit
thread: nudity
takeaway: >
  Re-edits exp061's 21 human-approved triples from their SAVED ORIGINALS with edit_latent_reflected
  and construction-derived masks — no GPU, no regeneration, no new human review, because the source
  clips are unchanged. Fixes 20 frozen single-donor targets and 7 wrong concept masks that together
  were 59% of exp080's training set. Not yet run.
---
# exp087 — re-edit exp061's triples with the reflected fill

## Why
exp080 collapsed concept motion to **-87% .. -99%** against a base of 0.686 in every arm of its LR
grid, while unrelated motion was untouched — a targeted freeze on concept prompts. The cause is in
the data: **20 of exp061's 21 human-approved triples have a `donor_map` of one repeated frame**
(`[7,7,7,7,7]`), because they were built 2026-08-02, three days before `edit_latent_reflected`
replaced `edit_latent`'s frozen single-frame fill. Those 21 are 59% of exp080's 34-triple dataset,
so the majority of its training targets literally encode "on a nudity prompt, emit a still image".
exp055 measured the same pathology at concept -84%.

A dry run also found a second defect nobody had counted: **7 of the 21 `concept_latent_mask`s
disagree with the construction mask.** They were detector-derived (exp061 predates the
mask-from-construction fix too), and with `nonfire_frame_weight: 0.0` the erase loss is *hard-masked*
to those frames — so for a third of the data the loss was being applied to the wrong frames
entirely.

**This costs no GPU and no re-review.** `frame_replace_split_precompute` saves the pre-edit clip as
`original_latent_path` alongside the edit, and only the edit depends on the edit rule. Re-editing
from the originals leaves the source clips byte-identical, so exp061's human approval of those 21
still holds. That is the difference between an hour and a regeneration plus a fresh review pass.

## Setup
`tools/reedit_frame_replace_dataset.py`, CPU-only. Per entry: load the original, rebuild both masks
from `split_latent_frame`/`concept_region` via the builder's own `build_edit_masks` (extracted and
now shared, so a re-edit cannot drift from how a fresh build would do it), apply
`edit_latent_reflected`, write the new edited latent and symlink the unchanged original.

Run it **where the latents live** — the cluster, like the merge step:

```
uv run python tools/reedit_frame_replace_dataset.py \
  --metadata experiments/exp061_split_nudity_dataset/metadata_human_filtered.json \
  --latents-dir experiments/exp061_split_nudity_dataset/outputs_20260802_223148/latents \
  --output-dir experiments/exp087_reedit_exp061_reflected/reedited
```

Expected: `21 entries ... frozen (single-donor) targets: 20 before -> 0 after ... concept masks
corrected to construction: 7`.

`--dry-run` reports the same counts without writing, and needs no latents.

**Detector fields are dropped, not carried over.** `edited_frame_confidences` /
`edited_max_confidence` describe the *old* edit and would be silently wrong against the new one;
rescoring needs a VAE decode (a GPU), which defeats the purpose. The detector is logging-only in
this pipeline — it gates nothing — so their absence costs nothing. Fields describing the original
clip are unchanged and carried through.

## What to watch
- The two counts above. `20 -> 0` frozen and `7` masks corrected are the whole point; anything else
  means the source metadata is not what we think it is.
- A visual check of two or three re-edited clips before training on them. The fill is mirrored
  motion rather than a still, so the edited half should *move* — that is the entire hypothesis, and
  per [[feedback-detector-metrics-not-ground-truth]] it should be watched, not inferred from the
  donor map.
- exp081 validated `edit_latent_reflected` on freshly built clips; this is the first time it has
  been applied retroactively to an existing dataset.

## Downstream
A training run merges these 21 with exp078's 13 (already clean — built 16 minutes after the fix
landed, donor maps like `[4,3,2,1,0,1,2,3]`) exactly as exp080's `combined_dataset/` was built, then
trains at whichever `(eta, retention set)` exp085/exp086 name. That run, not exp080, is the
candidate for the reported checkpoint; exp084 is then re-pointed at it.

**exp080, exp085 and exp086 are deliberately left alone.** They ran on the frozen data and stay
reproducible: exp080 is the finding that produced this diagnosis, and the eta grids measure whether
eta can work *around* donor overfitting — which frozen donors make maximally visible.

## Status
- [x] `tools/reedit_frame_replace_dataset.py` written; verified end-to-end on synthetic latents
      (mirrored fill, originals symlinked, stale keys dropped, training keys intact).
- [x] `build_edit_masks` extracted from the builder so both paths share one definition.
- [x] Dry run against exp061's metadata: 20 frozen, 7 masks to correct.
- [ ] Run on the cluster against the real latents.
- [ ] Two or three re-edited clips eyeballed.
- [ ] Merged with exp078's 13 into a training dataset.
