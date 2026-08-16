---
status: ready
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  exp070 rebuilt on repaired data: 27 rows at 16 first / 11 second (was 14 at 10/4) with the two
  blank targets removed, one of which had a 73%-blank edit that still contained a church. Answers the
  question exp070 could not: does church resist frame_replace because it is scene-level, or because
  its dataset was small, skewed and partly wrong? Blocked on exp123 for the eta. Not submitted yet.
---
# exp125 — church, second attempt

## Why exp070 does not settle the church question
It did not erase — top-1 oscillated 0.00 / 0.32 / 0.00 / 0.22 / 0.47 and trended *back* toward base,
alternating between destroyed clips (step 300: an abstract wooden plane over neon-green grass) and
clean substitutions (step 400: a barn in a meadow). Three candidate causes, and two of them are
artefacts of the data rather than statements about the concept:

1. **Size and skew** — 14 rows, 10 `first` / 4 `second`, against exp069's 21 balanced rows.
2. **A poisoned row** — `p4_s3305`, found on 2026-08-16 by exp122's degenerate-clip check: 36 of 49
   frames of its edited target are blank and the rest still show a church, so 7% of the erase signal
   was regressing toward "blank frames plus the concept". It passed the old screen *because* its blank
   half read as concept-free (exp122's notes have the general defect).
3. **Church is scene-level** — removing a frame-filling structure means redrawing the frame, where a
   chain saw can be swapped inside an untouched workshop. Not fixable by data.

This run removes 1 and 2 so that 3 can be read on its own.

## Setup
exp070 field-for-field except the dataset, the step schedule and the eta:

- **Dataset**: exp118's re-screened 13 rows + exp122's 14 = **27 rows, 16 first / 11 second**, no
  blank targets. Build with `merge_dataset.sh` on the cluster (command in the config header).
- **400 steps at `save_interval` 50** (exp070: 600 at 100). Checkpoint density is the point — exp070
  never showed a stable zero across two consecutive checkpoints, and at its resolution a transient one
  would have been invisible.
- **`erase_esd_eta` comes from exp123.** If that ablation finds no eta effect, keep exp070's 2.0 so
  this run isolates the data repair; if it does, use the winning arm. **Do not submit before exp123
  reports** — a rebuild that changes data *and* eta at once cannot answer either question.

## What to watch
- **Two consecutive checkpoints at top-1 0.00.** That is the bar exp070 never cleared, and the only
  evidence that would distinguish erasure from oscillation.
- **Top-5, which is the honest one.** exp070 reached 0.88 against a base of 0.95 at the same step
  where top-1 read 0.22 — the object was still fully there and the classifier's first choice had
  merely wobbled.
- **Colorfulness against base 64.** exp070 sat at 92-99 the whole run; that over-saturation is
  visible in the frames and is the scene-destruction signal for this class.
- **Whether erasure works by substituting one specific building.** exp118's substitutes were varied
  precisely to prevent this. Eyeball `eval_step_*/concept/` — a barn every time is a different
  (weaker) result from a varied absence.

## Reading it
| outcome | means |
|---|---|
| stable zero, scene intact | church erases; the pilot covers both a localized and a scene-level class and the remaining eight are worth running |
| still oscillating | frame_replace's success is concept-dependent — localized objects yes, scene-level no. That is a publishable limitation, not a bug, and it should be stated in `docs/comparison_targets.md` §2.2 as a measured result rather than a prediction |
| erases but destroys the scene | the erase and preserve pressures are irreconcilable for this class at this retention weight; `retention_weight` is then the next knob, not more data |

## Status
- [ ] exp123 reported; eta chosen and recorded here.
- [ ] `merge_dataset.sh` run on helios; row count asserted at 27.
- [ ] Submitted.
- [ ] Result written up against exp070 and exp069.
