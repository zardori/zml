---
status: done
concept: nudity
method: eval
thread: nudity
takeaway: >
  NegPrompt baseline for nudity, and it is strong: -68% on I2P (0.326->0.105) and -52% on
  SafeSora (0.480->0.230), both p<0.001, at NO measurable quality cost (DOVER flat, deltas
  inconsistent in sign) and near-zero collateral on unrelated prompts. The hoped-for 'it erases
  but wrecks generation' story did not happen. Our case rests on the residual: it still lets
  ~1 in 4 blunt video prompts through. Beat 0.105/0.230.
---
# exp083 — NegPrompt baseline (nudity)

## Why
A reviewer's first objection to any erasure method is that a training-free negative prompt might do
the same job for free. T2VUnlearning compares against exactly this baseline, and the ImageNet side
of this project already runs it (exp065, `negative_prompt: auto`). Nudity had no such row, so
"frame_replace erases nudity" had nothing cheap to be measured against. This fills that row for one
generation job per benchmark and no training.

If NegPrompt turns out to match us, that is far better learned now than from a reviewer in
September.

## Setup
Identical to exp082 — same base model, same two external benchmarks, same `(prompt, seed)` pairs,
same unrelated control — with exactly one difference: `negative_prompt` is set. So the A/B isolates
the intervention and nothing else. Grids over `control_concept_prompts` into run_001 (I2P) and
run_002 (SafeSora).

**Negative prompt**: `"nudity, naked, nude, bare skin, explicit sexual content"`.

Two deliberate choices worth defending in the paper:

- **Several synonyms, not the bare word "nudity".** A one-word negative prompt would be a weak
  baseline that we beat by under-specifying it; a real deployment would list synonyms. The ImageNet
  side negates just the class name because there a class name *is* the complete concept, whereas
  "nudity" has common near-synonyms. If anything this biases *against* us, which is the right
  direction for a baseline.
- **Applied to every prompt set, including `unrelated`.** NegPrompt is a deployed inference-time
  defence — it is on for all generations, not selectively for concept prompts. The collateral damage
  it does to unrelated prompts is therefore part of what is being measured, exactly as PSR captures
  it on the object side. Implemented in `zml/unlearn/eval.py:evaluate`; recorded as
  `_negative_prompt` in the run's `metrics.json` so the output dir is self-describing.

## What to watch
- **The tradeoff, not just the erasure number.** NegPrompt's known failure mode is that it degrades
  generation broadly — that is the whole reason trained erasure exists. The interesting row is
  concept detection *together with* unrelated `clip_score_mean`/`motion`/`colorfulness`. "NegPrompt
  erases as well as we do but wrecks unrelated prompts" is a perfectly good result for us;
  "NegPrompt erases as well and costs nothing" is a problem we need to know about now.
- Whether it behaves differently on the two prompt distributions (I2P's long art prompts vs
  SafeSora's short blunt ones) — CFG-based steering is sensitive to how much of the prompt is
  about the concept.
- Per [[feedback-detector-metrics-not-ground-truth]], a low NudeNet rate here needs the same visual
  spot-check we give our own runs before it is believed; do not hand NegPrompt a pass on the
  detector alone that we would not accept for frame_replace.

## Status
- [x] `negative_prompt` support added to the shared eval path (`eval_model.Config` +
      `zml/unlearn/eval.py`); verified it is a strict no-op when unset, so no prior run changes.
- [x] Config drafted; grid + `Config` construction verified end-to-end locally.
- [x] Submitted; all grid runs generated every clip, then timed out during scoring (see above).
- [x] Scored post-hoc with `tools/score_eval_videos.py` — no regeneration needed.
- [x] Compared against exp082 (base) on the same pairs; tradeoff column read, not just erasure.
- [x] DOVER filled in post-hoc (`tools/score_dover.py`, 2026-08-07) — flat, see below.
- [ ] Visual spot-check before believing the low detection rates.

## Results (2026-08-07) — a strong baseline that does not erase

| set | n | base (exp082) | NegPrompt | change |
|---|---|---|---|---|
| I2P | 95 | 0.326 | **0.105** | -68% rel, z=3.70 |
| SafeSora | 100 | 0.480 | **0.230** | -52% rel, z=3.69 |

Both drops are significant well past p<0.001, so **NegPrompt is not a strawman** — the strong
multi-synonym form we chose does substantial work for free, and any erasure claim we make has to
beat 0.105 / 0.230, not 0.326 / 0.480.

**But it does not erase.** Residual nudity is 10.5% (+/-6.2pp) on I2P and 23.0% (+/-8.2pp) on
SafeSora. Roughly a quarter of blunt video-native nudity prompts still produce nudity with an
explicit five-synonym negative prompt in place. That is the gap a trained method exists to close.

### The cost is real, and CLIP score does not see it

| set | clip | colorfulness | motion |
|---|---|---|---|
| I2P concept | 0.260 -> 0.245 (-5.5%) | 38.07 -> 53.57 (**+40.7%**) | 0.779 -> 1.580 (**+102.8%**) |
| SafeSora concept | 0.275 -> 0.257 (-6.6%) | 37.54 -> 44.92 (+19.7%) | 1.662 -> 2.393 (+44.0%) |
| unrelated | 0.331 -> 0.329 (-0.5%) | 33.76 -> 39.23 (**+16.2%**) | 2.013 -> 2.189 (+8.7%) |

Two things fall out of this table.

**NegPrompt is not surgical — it moves the whole output distribution.** Colorfulness rises 41% and
motion doubles on the concept set. Steering CFG away from "nudity, naked, nude, bare skin, explicit
sexual content" does not subtract the concept and leave the rest; it produces materially different
video, with prompt adherence down 5-7%.

**It is not free on prompts that never contained the concept.** The `unrelated` set is 15 prompts
with no nudity in them, generated from identical seeds, and colorfulness still moves 16%. This is
the collateral column exp083 was built to produce — and note that **CLIP score would have reported
"no collateral damage" (-0.5%)**. Had we tracked only clip_score we would have concluded NegPrompt
was cost-free and been wrong. Report the visual statistics alongside it.

### DOVER settles it: the distribution shift is not quality damage

Scored post-hoc on x86_64 with `tools/score_dover.py` (helios reports 0.0 — aarch64):

| set | technical | aesthetic |
|---|---|---|
| I2P concept | 0.0826 -> 0.0873 (+5.7%) | 0.9296 -> 0.9193 (-1.1%) |
| SafeSora concept | 0.0900 -> 0.0868 (-3.6%) | 0.9413 -> 0.9481 (+0.7%) |
| unrelated | 0.0878 -> 0.0903 (+2.8%) | 0.9666 -> 0.9647 (-0.2%) |

Every delta is small and **the signs are inconsistent** — technical rises on I2P and falls on
SafeSora, aesthetic does the reverse. That is noise, not an effect. Base quality is also flat across
all three sets, so the model is not producing worse video on nudity prompts to begin with.

So NegPrompt's +41% colorfulness, +103% motion and 2x H.264 bitrate are **genuinely different but
technically sound video**, not artefacts. The bitrate already ruled out the opposite failure
(collapse to blank/static output compresses to nearly nothing); DOVER rules out the remaining one.

**This is the harder result for us, and it overturns the expectation this experiment was set up
with.** The hoped-for story was "NegPrompt erases as well as we do but wrecks generation" — that is
what the What-to-watch section above anticipated. It did not happen. NegPrompt buys 52-68% erasure
at no measurable quality cost and near-zero collateral on unrelated prompts (clip -0.5%, DOVER
flat). The honest summary is that it is a **strong, cheap baseline with a modest fidelity cost**
(-5.5%/-6.6% clip on concept prompts, i.e. you get sound video that is less what you asked for).

### What our method has to do

**Beat 0.105 (I2P) / 0.230 (SafeSora) on erasure.** That is now the whole argument, and it is a
sufficient one: a deployed five-synonym negative prompt still lets roughly **one in four** blunt
video-native nudity prompts through. Closing that residual is what a trained method is for.

Two secondary claims remain available and are worth measuring rather than assuming:
- **Equal or better quality** — cheap to show now that the DOVER numbers exist for the base and
  NegPrompt rows on identical prompts and seeds.
- **Better preservation on nudity-adjacent content.** Neither baseline is tested on this at all;
  `unrelated` is generic content. A held-out `related` set is the only place a method that erases
  *without* destroying swimwear can be distinguished from one that does not — and per exp079 our own
  detector scores a red bikini at 0.844, so the concept column alone would reward the wrong model.

### Still unsettled

Per [[feedback-detector-metrics-not-ground-truth]], none of the above is a substitute for looking at
clips. DOVER is a learned quality model, not ground truth, and the residual-nudity clips deserve the
same scepticism in the other direction.

## Run 1 (2026-08-07): timed out in scoring — videos intact, no regeneration needed

All grid runs hit `CANCELLED ... DUE TO TIME LIMIT` at the 6h budget. An eval job generates on GPU
first and scores on CPU last, so a too-small budget kills it *after* the expensive part. Every clip
was written before the kill — 95/95 (I2P), 100/100 (SafeSora), 15/15 unrelated in each run, verified
on the cluster — and only `metrics.json` is missing.

**Two causes, not one** — see exp082's notes for the full account: the 6h budget was too small, but
the scoring tail was also ~9x slower than it should have been because NudeNet let onnxruntime size
its thread pool to helios' 288 cores. Fixed in `zml/benchmarks/check_for_nudity.py`, scores
unchanged.

So the ~24 h of GPU work is intact and does **not** need repeating. Recover with
`tools/score_eval_videos.py` (added for this), which reads the run's own config to pair each clip
with the prompt that generated it and writes the same `metrics.json` via the same `score_video_dir`
the eval path itself uses — a recovered run is not scored differently from a normal one:

```
uv run python tools/score_eval_videos.py <grid_dir>/run_001 <grid_dir>/run_002
```

It needs the videos, which are still cluster-side (the last pull excluded them). Run it on the
cluster, or pull those `eval_step_0/` dirs first. DOVER only contributes on x86_64, so if scored on
helios its fields stay 0.0 and can be filled later with `tools/score_dover.py`.

`slurm_time` raised 6h → 12h so a resubmission does not repeat this.
