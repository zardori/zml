---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  THE REPORTED 2B ROW, AND IT FALLS SHORT OF THE BAR. Restricted (10-way) convention: ESR-1 49.90,
  ESR-5 15.61, PSR-1 82.71, PSR-5 93.19 against GOAL.md's target/guards (92.38 / 77.09 / 54.03 /
  82.14) — ESR-1 and ESR-5 miss badly, PSR-1 and PSR-5 clear their floors with room to spare. The
  erased-class motion guard also passes (0.390 vs the 0.15 floor; exp130's base was 0.840). Nearly
  identical to exp071's 5b row under the same eta=2.0 recipe (restricted ESR-1 49.0 / ESR-5 10.0,
  PSR-1 83.8) despite a different, larger training set (33 rows on 5b vs 25 here) — so the ceiling
  looks like a property of eta=2.0 erase pressure, not of model scale or this dataset's size.
  Restricted top-5 on chain saw itself barely moves (0.844 vs base 1.0), so most of the erasure is a
  #1-guess demotion, not removal — the object still shows up in the top-5 almost as often as before.
  exp133's live-monitor read that the freeze is not concept-conditional (unrelated motion *rising*
  35%) does NOT hold on the full protocol: computed against exp130's per-class base motion, the nine
  preserved classes lose a mean ~32% of their motion (French horn -69%, garbage truck -52%, cassette
  player -49% worst; English springer actually +7% best), the same small-sample-optimism failure
  exp071 found on 5b (there ~45% mean loss), just less severe. Motion guard still passes at every
  class. Unblocks exp135: an erase_esd_eta sweep above 2.0 (never tried on either model) to test
  whether more erase pressure closes the ESR gap before the motion margin runs out.
---
# exp134 — reported ESR/PSR for the chain-saw LoRA, on CogVideoX-2B

## Why
GOAL.md's target table is CogVideoX-2B, restricted (10-way) convention, compared against
T2VUnlearning's Table 4. exp133 trained the first 2B chain-saw `frame_replace` LoRA and its
9-prompt live eval reproduced exp069's 5b trajectory (concept top-1 0.09 → 0.00 by step 200,
holding through step 600). But exp071 already showed once, on 5b, that a small live-eval set can
mislead — exp069's monitor read the motion freeze as concept-conditional; the full 200-prompt
protocol showed it was global (nine preserved classes losing a mean 45% of their motion). exp133's
live eval shows the *opposite* surprise this time (unrelated-sample motion rising, not collapsing)
and that needs the same correction: only the full protocol tells us if it holds.

This run is the 2B counterpart of exp071 — same config shape, same checkpoint-selection logic,
new model_id and new `lora_checkpoint_dir` pointing at exp133 instead of exp069. It produces the
row that gets checked against GOAL.md's target table and every guard in it.

## Hypothesis and what would falsify it
Hypothesis: 2B's ESR/PSR row lands close to exp071's 5b row in shape — strong ESR-1/ESR-5 under
the 1000-way convention, a smaller but real gain under restricted, PSR held close to exp130's 2B
`Original` (restricted PSR-1 89.40, PSR-5 97.91) — and the GOAL.md motion guard (0.15 floor on the
erased class) either passes or fails the same way 5b's did (5b's own erased-class motion was 0.111,
already below the 0.15 floor exp071 helped calibrate).

Falsified by: restricted ESR-1/ESR-5 not clearing T2VUnlearning's bar (92.38/77.09) — expected on
a single checkpoint with no hyperparameter search, so not itself a failure of the method, just a
result to report honestly. More load-bearing: if the full-protocol preserved-class motion mean
matches 5b's collapse (~-45%) despite exp133's live sample suggesting otherwise, that confirms the
freeze is model-independent, not something 2B improved on; if it does NOT collapse, that is a
genuine 2B-specific finding worth its own note in `docs/imagenet_objects.md`.

## Setup
Field-for-field exp071 except:
- `model_id: THUDM/CogVideoX-2b`
- `lora_checkpoint_dir` points at exp133's final checkpoint
  (`experiments/imagenet/exp133_frame_replace_chainsaw_2b/outputs_20260820_181735/frame_replace_lora_step600`)
  instead of exp069's.
- `slurm_time` raised to match exp071's 14h (see config.yaml comment): exp130 measured 2B base-only
  generation at 4.6h, but applying exp071's own base-to-LoRA ratio (2.4x) projects close to a 10h
  cap, and LoRA overhead is not guaranteed to shrink with model size.

Everything else — 200 prompts, 10 classes, `erased_class: "chain saw"`, 50 inference steps,
`disable_mlflow` — is unchanged, so the row is comparable to exp071's under both the 1000-way and
restricted conventions, and to exp130's `Original` 2B row for the PSR delta.

## What to watch
- **Restricted ESR-1 / ESR-5 / PSR-1 / PSR-5** against GOAL.md's target table (92.38 / 77.09 / 54.03
  / 82.14) and its four guards, including the motion floor (0.15).
- **ESR-5 specifically**: T2VUnlearning's own claim is that baselines inflate ESR-1 while leaving
  ESR-5 low (distortion, not removal). exp071 showed our 5b row does this too under the restricted
  convention (ESR-1 49.0, ESR-5 10.0) despite passing outright under 1000-way (100.0 / 89.8). Watch
  for the same split here.
- **Per-class motion on the nine preserved classes**, against exp133's live-sample surprise (rising,
  not falling). This is the check exp133's notes explicitly deferred to this run.
- **Colorfulness and DOVER** on the erased class and the nine preserved ones, same over-saturation
  signal exp071 found on all ten 5b classes.

## Results (2026-08-21) — reported row, misses the ESR bar, PSR and motion guards pass

Completed on athena in 5.3h (job 3020535, of a 14h budget).

Restricted (10-way) — the convention compared against GOAL.md's table:

| | ESR-1 | ESR-5 | PSR-1 | PSR-5 |
|---|---|---|---|---|
| Target/guards | ≥92.38 | ≥77.09 | ≥54.03 | ≥82.14 |
| exp134 (2B) | 49.90 | 15.61 | 82.71 | 93.19 |
| exp071 (5b) | 49.0 | 10.0 | 83.8 | ~93 |

ESR-1 and ESR-5 both miss by a wide margin; PSR-1 and PSR-5 both clear their floors comfortably.
The row is almost a repeat of exp071's 5b number under the identical eta=2.0 recipe, despite 5b
training on a larger, differently-sourced 33-row set (exp126's exp066+exp117+exp121 merge) against
this run's 25-row exp131 set — model scale and dataset size both differ, the ESR number does not.
That points at `erase_esd_eta` (fixed at 2.0 in every run since exp126, never swept higher) as the
more likely ceiling, not either of those two variables.

Chain saw restricted top-1 goes 0.8847 (exp130 base) → 0.5010 (this run), top-5 1.0 → 0.8439 — a
real drop, but far short of the near-total top-1 collapse the live 9-prompt training monitor
suggested (0.09 → 0.00 by step 200, holding to step 600). Top-5 staying near 0.84 means the object
is still recognized in the top-5 guesses on most clips; erasure is mostly demoting it from #1, not
removing it from the frame.

**Motion — the deferred check from exp133.** Guard passes: erased-class motion is 0.390, well above
the 0.15 floor (base 0.840, exp130). But exp133's live-monitor read (unrelated-sample motion RISING
35%, the opposite of 5b's global collapse) does not survive contact with the full protocol. Computed
here against exp130's per-class base motion, the nine preserved classes lose a mean ~32%: French
horn -69%, garbage truck -52%, cassette player -49% worst three; golf ball -42%, church -36%, gas
pump -31%, tench -1%, parachute -13%, English springer the only class to gain (+7%). So the freeze
is *not* concept-conditional at 2B either — same qualitative finding as exp071 on 5b (there: ~45%
mean loss) — just measurably less severe, and still inside the guard at every class. exp133's
9-prompt live sample was optimistic in the same way exp069's was, just in the opposite direction
(it undersold the loss instead of overselling the freeze).

Colorfulness: not checked against exp130's base per-class in this pass (deferred — motion and the
ESR/PSR gap were the load-bearing questions this run answered).

## Status
- [x] Submitted (athena job 3020535, completed 2026-08-21T07:52).
- [x] Row measured under both conventions; checked against GOAL.md's target table and all four
      guards — ESR-1/ESR-5 fail, PSR-1/PSR-5 and the motion floor pass.
- [x] Per-class motion on the nine preserved classes checked against exp133's live-sample reading —
      the live sample was misleading (mean ~32% loss on the full protocol vs a live-sample rise),
      same failure mode as exp071 on 5b, smaller magnitude.
