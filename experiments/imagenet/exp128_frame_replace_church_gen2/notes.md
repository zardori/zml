---
status: done
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  THE DATA WAS NOT THE PROBLEM. On the repaired 27-row set (16 first / 11 second, blank targets
  removed) church still does not erase — top-1 oscillates 0.11 / 0.40 / 0.22 / 0.22 / 0.18 / 0.21 /
  0.40 / 0.34 across all eight checkpoints and never once reaches 0.00, where exp070's smaller,
  skewed, partly-poisoned set at least touched it twice. Top-5 holds 0.53-0.89 against base ~0.95, so
  the object is present throughout, and colorfulness climbs to 102.5 against base 64 — the
  scene-destruction signal, worse than exp070's 92-99. Doubling the data, fixing the positional skew
  and removing the poisoned row changed nothing, which rules out causes 1 and 2 and leaves cause 3:
  church resists frame_replace because it is scene-level. With exp069 this makes the method's
  concept-dependence a measured result, not a prediction.
---
# exp128 — church, second attempt

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
- **`erase_esd_eta` comes from exp126.** If that ablation finds no eta effect, keep exp070's 2.0 so
  this run isolates the data repair; if it does, use the winning arm. **Do not submit before exp126
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

## Results (2026-08-17) — repaired data, same failure

Completed in 8.9 h on helios, 400 steps, `save_interval` 50, `erase_esd_eta` 2.0. Base for this class
is top-1 **0.733**, top-5 ~**0.95**, colorfulness **64**.

| step | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
|---|---|---|---|---|---|---|---|---|
| top-1 | 0.11 | 0.40 | 0.22 | 0.22 | 0.18 | 0.21 | 0.40 | 0.34 |
| top-5 | 0.56 | 0.75 | 0.69 | 0.69 | 0.53 | 0.56 | 0.89 | 0.71 |
| colour | 84.8 | 88.2 | 84.0 | 98.3 | 99.4 | 100.2 | 102.5 | 88.6 |
| motion | 0.030 | 0.065 | 0.048 | 0.044 | 0.014 | 0.054 | 0.033 | 0.052 |
| unrelated motion | 0.332 | 0.535 | 0.594 | 0.470 | 0.520 | 0.517 | 0.568 | 0.509 |

### The bar it had to clear, and did not

The pre-registered read was **two consecutive checkpoints at top-1 0.00** — the thing exp070 never
showed. This run shows **zero checkpoints at 0.00**, at twice exp070's checkpoint density. So the
denser sampling did not reveal a hidden stable window; there is not one.

It is, if anything, *worse* than exp070, which hit 0.00 at steps 100 and 300. Reading that as a
regression would over-interpret an oscillating signal — the honest statement is that both runs
oscillate in the same 0.1–0.5 band with no trend, and the repaired data did not move the band.

### Top-5 is the honest column, and it never moves

Top-5 runs 0.53–0.89 against a base of ~0.95, and its highest value (0.89, step 350) sits at a step
where top-1 also reads 0.40. The classifier's first choice wobbles; the church does not leave. This
is the same pattern flagged in "What to watch" from exp070's 0.88.

### The scene is being destroyed, not edited

Colorfulness climbs monotonically from 84.8 to 102.5 through step 350 — **60% over base** and above
exp070's 92–99 range. Concept motion sits at 0.014–0.065 while the unrelated set holds 0.33–0.59, so
as with exp069 the damage is concept-conditional; unlike exp069 it buys no erasure in exchange.

### What this rules out

The three candidate causes from the header:

1. **Size and skew** — removed: 27 rows at 16/11 against 14 at 10/4. No change.
2. **The poisoned row** (`p4_s3305`, 36/49 blank frames in its edited target) — removed. No change.
3. **Church is scene-level** — the remaining explanation, now by elimination rather than by assertion.

That is outcome row 2 of "Reading it": **frame_replace's success is concept-dependent — localized
objects yes, scene-level no.** Per that row this is a publishable limitation and belongs in
`docs/comparison_targets.md` §2.2 as a measured result.

### A caveat on the eta gate

The header said "do not submit before exp126 reports". exp128 was in fact submitted alongside exp126
(both 2026-08-16 ~18:20), not after it. It landed on `erase_esd_eta: 2.0` — exp070's value, and the
one exp126 went on to endorse ("keep 2.0") — so the data-repair comparison is clean and the confound
the gate existed to prevent did not occur. Worth noting because the gate was bypassed, not satisfied.

## Status
- [x] exp126 reported; eta 2.0 confirmed as the right choice (though in parallel, see caveat above).
- [x] `merge_dataset.sh` run on helios; 27 rows.
- [x] Submitted and complete (1 job, 8.9 h on helios).
- [x] Result written up against exp070 and exp069.
- [ ] Eyeball `eval_step_*/concept/` frames — is the failure a varied absence or a repeated substitute
      building? exp070 showed both a degenerate plane and a clean barn; this run's frames are unseen.
- [ ] `docs/imagenet_objects.md` and `docs/comparison_targets.md` §2.2 updated with the limitation.
- [ ] Decide whether `retention_weight` (outcome row 3's knob) is worth one more job, or whether the
      scene-level limitation is the result we report.
