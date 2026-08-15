---
status: ready
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  Second generation of the church dataset — exp118's prompts verbatim under seeds 3331-3360. Adds
  ~14 rows at exp118's measured 47%, and is the only way to fix that set's 10-first / 4-second
  region skew without discarding rows. Not submitted yet.
---
# exp122 — church dataset, generation 2

## Goal
Same as exp121 (see it for the shared reasoning): sample exp118's measured 47% again rather than
edit prompts further.

## What is church-specific
exp118's 14 survivors split **10 `first` / 4 `second`**. `concept_region` is resolved per seed, so
the only way to rebalance without throwing away good clips is to draw more seeds. If gen2 comes back
similarly skewed, that is not bad luck twice — it would mean something about the church construction
favours one side, and the fix is `concept_region: second` on a third build rather than more sampling.

Also worth carrying forward: two of exp118's 30 clips were degenerate white frames (`p9_s3310` pure
white, `p21_s3322` near-white, spatial std 0.0 and 11.8). Both fell out on the concept screen, so
nothing was poisoned, but check gen2 for the same — a blank clip that happened to pass would be the
worst possible training target.

## What to watch
- Pass count against exp118's 14/30, and `not-split` against its 5 (the substitute rewrite's number).
- **Survivor region balance**, per above.
- Degenerate low-variance clips.

## Downstream
Merge with exp118's screened set for the second church training run. exp070 trains on exp118 alone.

## Status
- [ ] Submitted.
- [ ] Screened (`tools/screen_split_dataset.py --min-concept-max 0.10 --write-filtered`).
- [ ] Pass count and region balance compared against exp118.
- [ ] Checked for degenerate blank clips.
- [ ] Merged with exp118 for the follow-up run.
