---
status: done
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  Did its job twice over. 14/30 usable at a 7 first / 7 second balance — the exp118 skew is fixed, and
  the merged church set is 27 rows / 16 first / 11 second. And its degenerate-clip check found a
  defect in the SCREEN, not the build: p22_s3353 passed with a near-perfect contrast index (+0.994)
  while its edited target is 49/49 blank frames, because a safe half that never rendered reads as
  concept-free. tools/screen_split_dataset.py now gates on the edited target's blank-frame share,
  which also retired exp118's p4_s3305 — a 73%-blank target that WAS in exp070's training set.
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

## Results (`outputs_20260816_002006`, helios, 1 h 39 m)

| | pass | not-split | no-concept | blank-target | first / second |
|---|---|---|---|---|---|
| exp118 (reference, re-screened) | 13/30 (43%) | 5 | 10 | 2 | 9 / 4 |
| **exp122** | **14/30 (47%)** | 4 | 10 | 2 | **7 / 7** |

- **The skew is fixed.** 7/7 here, and merged with exp118 the church set is **27 rows, 16 first /
  11 second** — from 10/4 on exp070's data. Nothing had to be discarded to get there, which was the
  whole argument for drawing seeds instead of forcing `concept_region: second`.
- Yield reproduces exp118 (47% vs 43% after re-screening), the same seed-control result as exp121.

### The blank-clip check found a hole in the screen, not just a bad clip

Checking for degenerate clips (this experiment's fourth status box) turned up `p22_s3353`: contrast
index **+0.994**, concept max 0.4386 — a textbook pass — whose edited target is **49/49 blank
frames**. The source clip is 24/49 blank, all of it in the safe half.

The mechanism is general and applies to every concept, not just church. The screen's differential is
computed on the *source* clip's per-frame confidences, and a blank frame scores p(concept) ≈ 0 exactly
like a legitimately concept-free one. So a clip whose safe half never rendered gets a near-perfect
separation score, passes, and is then edited by mirroring that blank half into the concept region —
the better the blankness, the better the score. **The screen never looked at the target training
consumes.**

Re-screening the older builds with the new gate found one more, and it matters: **exp118's `p4_s3305`
— 16/49 blank in the source, 36/49 in the target, and the visible remainder is a church, i.e. the
"concept-removed" target still contains the concept. It was 1 of exp070's 14 training rows**, so 7% of
that run's erase signal was pointing the wrong way. Not the whole explanation for exp070's
oscillation, but a real contributor and now removed.

Fix landed in `tools/screen_split_dataset.py`: a `blank-target` verdict on the edited clip's
blank-frame share (`zml/benchmarks/frame_quality.py::degenerate_frame_mask`, `MAX_DEGENERATE_FRAC`
0.1), checked *before* the concept gates so the diagnosis is not hidden behind a "pass". It needs the
videos next to the metadata and warns loudly when they are absent. Chain saw is unaffected — exp066,
exp117 and exp121 have zero blank targets between them.

## Downstream
exp128 trains on this merged with exp118's re-screened set (27 rows). exp070 trained on exp118 alone,
pre-fix.

## Status
- [x] Submitted; completed 2026-08-16 (job 20735918).
- [x] Screened → `outputs_20260816_002006_screened.json`, 14 entries.
- [x] Pass count and region balance compared against exp118.
- [x] Checked for degenerate blank clips — found two, and a screen defect behind them.
- [ ] Merged with exp118 for exp128 (`merge_dataset.sh`, run on the cluster).
