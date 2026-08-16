---
status: done
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  NEGATIVE RESULT, and the contrast that gives exp069 its meaning: in exp069's identical regime,
  church does not erase. Concept top-1 oscillates 0.00 / 0.32 / 0.00 / 0.22 / 0.47 and TRENDS BACK
  toward base (0.739) while top-5 reaches 0.88; the run is swinging between destroyed and unerased
  rather than converging. Frames show both states directly — step 300 is a degenerate wooden plane
  over neon-green grass (colorfulness 98 vs base 64), step 400 a clean barn substitution. So the
  method's success is not concept-independent: localized objects erase, a frame-filling scene-level
  class does not, at 14 rows skewed 10 first / 4 second — one of which was later found to be a
  73%-blank target that still contained a church. Rebuilt as exp125 on the 27-row exp118+exp122 merge
  (16/11, blank targets removed). Do not spend exp072's 200-video eval on this checkpoint.
---
# exp070 — frame_replace erasure of "church"

## Goal
The hard half of the pilot, run in exactly exp069's regime so the two are directly comparable. A
church fills the frame and defines the scene; a chain saw sits inside one. Comparing the two isolates
how much frame_replace depends on the concept being *localized*, which is the property
`docs/comparison_targets.md` §2.2 claims makes objects its native regime.

Reference point: T2VUnlearning's per-class ESR-1 is 100 on garbage truck and French horn but 82.35 on
church — the class every method finds hardest.

## Setup
Identical to exp069 apart from the dataset, `concept_target`, `retention_exclude` and the control
prompt files.

**Dataset: exp118's 14 screened rows**, no merge. exp067 run 2's 3 survivors are excluded on purpose:
all three are `concept_region: first`, and exp118's set is already 10 first / 4 second — three more
would take it to 13/4 for a 21% size gain. exp122 draws fresh seeds to rebalance instead. (This is
the opposite call from exp069, where the older build's rows both balanced the sides and added the
framing diversity church does not need — its clips are all wide by nature.)

**Read the skew, not around it.** A 10/4 keep set can be satisfied by the positional shortcut "copy
the concept-free half onto the other". The 20 church eval prompts are ordinary full scenes with no
object-free half, so they cannot be lowered by a shortcut LoRA — which makes the concept-set curve
the shortcut test, exactly as designed.

`./submit_job.py helios experiments/imagenet/exp070_frame_replace_church/config.yaml`

## What to watch
- Same three reads as exp069 (erasure, shortcut, collateral), plus:
- **What replaces the church.** If erasure works by substituting a specific building (the B-prompt
  substitutes were varied precisely to avoid this), that shows up as coherent buildings in the eval
  videos rather than absence. Worth eyeballing `eval_step_*/concept/`.
- **Scene damage.** Removing a frame-filling structure risks taking the surrounding scene with it —
  watch clip score and colorfulness on the unrelated set more closely than in exp069.

## Results (`outputs_20260816_005801`, helios, job 20737525) — church does not erase

The job ran into its 16 h wall at step ~550 (95 s/step against exp069's 66 s/step, so 600 steps needed
~15.8 h; `run_info.json` still reads `running` because it never got to write its epilogue). Steps 100
through 500 are complete and they decide the question, so this is not being resubmitted for the last
50 steps.

| step | top-1 | top-5 | clip | colorfulness | motion | DOVER tech |
|---|---|---|---|---|---|---|
| base | 0.739 | 0.950 | 0.335 | 64.0 | 0.481 | — |
| 100 | 0.00 | 0.22 | 0.27 | 92 | 0.040 | 0.055 |
| 200 | 0.32 | 0.47 | 0.29 | 95 | 0.030 | 0.059 |
| 300 | 0.00 | 0.48 | 0.27 | 98 | 0.060 | 0.072 |
| 400 | 0.22 | **0.88** | 0.30 | 99 | 0.080 | 0.074 |
| 500 | **0.47** | 0.78 | 0.30 | 86 | 0.090 | 0.087 |

Base for church is top-1 0.739 / top-5 0.950, clip 0.335, colorfulness 64.0, motion 0.481, DOVER
technical 0.101. DOVER was backfilled locally (helios writes 0.0); note it *rises* as the run
progresses and is lowest at step 100, the one checkpoint where top-1 read 0.00 — another instance of
erasure and technical quality moving together in this thread.

**1. No erasure — and the trend runs the wrong way.** Top-1 never holds a zero for two consecutive
checkpoints, top-5 climbs to 0.88 against a base of 0.95, and by step 500 top-1 is back to 0.47 of
base's 0.739. Training loss meanwhile fell normally (0.476 → 0.247), which is exactly the
"loss looks fine either way" case `docs/split_prompt.md` §4 warns about.

**2. The two states are visible in the frames.** At step 300 (`concept/video_0.mp4`) the clip is an
abstract wooden plane over neon-green grass — that is what colorfulness 98 vs base 64 is measuring,
and it is scene destruction, not erasure. At step 400 (`video_1.mp4`) the model renders a plausible
barn in a meadow, a clean substitution with the church gone. The run is alternating between the two
rather than converging on the second.

**3. Why, in order of confidence.** (a) 14 rows, skewed 10 first / 4 second — half exp069's data with
a positional bias on top. (b) **One of those 14 rows was actively wrong.** exp122's degenerate-clip
check (2026-08-16, after this run) found `p4_s3305`: 36 of 49 frames of its edited target are blank,
and the rest still show a church. So 7% of the erase signal was regressing toward "blank frames, plus
the concept we are erasing". It passed the old screen precisely *because* its blank half read as
concept-free — see exp122's notes for the general defect and the fix. (c) Church is scene-level:
removing it means redrawing the frame, so the erase direction and the "keep the scene" pressure fight,
where a chain saw can be swapped inside an untouched workshop. This is the property
`docs/comparison_targets.md` §2.2 predicted would matter, and the pilot has now measured it:
**the method's success is concept-dependent.**

**4. Read against exp069.** Same recipe, same eta, same retention anchors, same step budget; chain saw
goes to a flat 0.00 and church does not move. Everything except the concept and its dataset is held
fixed, so this pair is the pilot's actual finding.

## Downstream
- **exp125** — rebuild on the exp118 + exp122 merge (27 rows, 16 first / 11 second, blank targets
  removed) at whichever eta exp123 picks. All three suspected causes are addressed at once, which is
  the point: if the rebuild still oscillates, the cause is scene-level-ness and not data.
- **exp072** stays blocked. Reporting a 200-video ESR/PSR row for a checkpoint that is 0.47 top-1 on
  the monitor would spend ~14 h of athena to publish noise.

## Status
- [x] Datasets complete; config wired to exp118's screened set and exp068's anchors.
- [x] Submitted; hit the 16 h wall at step ~550 with evals 100–500 complete.
- [x] Result written up; superseded by exp125 for the erasure question.
