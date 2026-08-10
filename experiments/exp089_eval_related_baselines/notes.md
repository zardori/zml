---
status: done
concept: nudity
method: eval
thread: nudity
takeaway: >
  Base and NegPrompt rows on the new held-out nudity-adjacent set
  (prompts/cogvideox_nudity_control_related.csv, 36 prompts). exp082/exp083 never generated a
  related column, so without these a preservation number from our checkpoint has nothing to sit
  beside. 72 clips total. RESULT (frame rate, the metric to read): base 0.149, NegPrompt 0.074 —
  NegPrompt HALVES near-miss content, which is real collateral damage. An earlier reading of this
  run called the set too narrow to discriminate; that was the video rate (0.056 vs 0.028) and it
  understated the signal — at frame level there is 15% baseline to work with. Its `concept` row is
  all zeros with total_videos null (the zero-filled-row bug, fixed after this run) — ignore it.
---
# exp089 — base and NegPrompt on the held-out `related` set

## Why
The results table has erasure, fidelity, distribution and quality for base (exp082) and NegPrompt
(exp083) on two external benchmarks. It has **no preservation column at all** for nudity: the only
nudity-adjacent prompts in the repo were exp079's *training* retention anchors, and scoring on those
after training against them is scoring on the training set.

`prompts/cogvideox_nudity_control_related.csv` (new) fills that: 36 held-out prompts across the same
nine categories as exp079 (swimwear, athletic, medical, sleepwear, bathing, parenting, clothed
intimacy, close crops, multi-person), 4 each, seeds 602001-602036, **zero prompt overlap** with the
training anchors. Mirrors fire's own two-set arrangement — `cogvideox_fire_preservation.csv` is
trained on, `cogvideox_fire_control_related.csv` is eval-only, and the two share nothing.

**Why this is now the load-bearing column.** exp083 killed the argument we expected to make:
NegPrompt buys 52-68% erasure at *no* measurable quality cost (DOVER flat, clip -0.5% on unrelated),
so "we erase with less damage" is not available on quality grounds. What remains is the residual
(NegPrompt still lets ~1 in 4 blunt prompts through) and **preservation of nudity-adjacent content**,
which neither baseline has ever been measured on.

And for nudity specifically this column inverts the usual reading. exp079 found NudeNet scores its
own clothed anchors as nudity — **0.844 on a red bikini across all 49 frames**. So on this set the
base model should have a *substantial* detection rate, and preservation means **keeping** it. A
model that erases by destroying swimwear appears here as a detection-rate **drop** — a failure our
headline concept metric would otherwise reward. That asymmetry is the strongest single argument for
reporting `related` at all, and it is why this cannot be inferred from the concept column.

## Setup
Base model, no `lora_checkpoint_dir`. Grids `negative_prompt` into both baseline arms in one config:

- **run_001 — `""`.** diffusers does `negative_prompt = negative_prompt or ""`
  (`pipeline_cogvideox.py:300`), so an empty string is byte-identical to passing nothing. Plain base
  model, and it is recorded as `_negative_prompt: ""` in `metrics.json` rather than silently absent.
- **run_002 —** the exp083 negative prompt verbatim, so this arm matches that experiment exactly.

36 prompts x 2 arms = 72 clips. `control_related_prompts` is the only set given; `Config.__post_init__`
requires at least one, and `include_related` is set automatically when the field is present.

## What to watch
- **The base detection rate should be high**, not low. If it comes back near 0.00 the set is not
  actually near-miss content and it will not detect over-erasure — the whole point. exp079's own
  clips give the expectation: 13 of 30 tripped the detector, topped by 0.844.
- **`clip_score_mean` alongside it.** The detector answers "does it still look like skin"; clip
  answers "is it still the scene we asked for". A model that preserves the detection rate by
  producing generic beach footage has not preserved anything.
- **Whether NegPrompt already damages this column.** It is applied to every prompt at inference, so
  it plausibly suppresses swimwear too. If it does, that is a preservation cost we can report
  against it — the differentiator exp083 took away on quality grounds.
- Per-category breakdown (the `category` column rides along in the CSV): swimwear and closeup are
  where over-erasure should bite first.
- Per [[feedback-detector-metrics-not-ground-truth]], and doubly here since the detector's behaviour
  on this content is the very thing being exploited: a human pass over the base clips is needed
  before the column is trusted, exactly as exp079 needed one.

## Downstream
Our own checkpoint's related row comes from **exp084**, which now carries the same CSV — same
prompts, same seeds, same detector — so the three rows are directly comparable.

## Status
- [x] `prompts/cogvideox_nudity_control_related.csv` written and validated (36 rows, 9 categories x 4,
      no seed collisions anywhere in `prompts/`, zero overlap with exp079's anchors).
- [ ] Submitted.
- [ ] Base detection rate confirmed substantial (else the set does not do its job).
- [ ] Human pass over the base clips.
- [ ] Compared against exp084's related column.
