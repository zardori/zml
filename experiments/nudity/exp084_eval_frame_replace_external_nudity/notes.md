---
status: done
concept: nudity
method: eval
thread: nudity
takeaway: >
  The method row. exp080 run_002 step 120 (the best point human review found) evaluated on I2P and
  SafeSora — same prompts, seeds and unrelated control as exp082 (base) and exp083 (NegPrompt), so
  the three form one table. RESULT, frame rate (the metric to read): I2P 0.0054 (25/4655 frames),
  SafeSora 0.0916 — against NegPrompt's 0.137 / 0.263 and base 0.346 / 0.500. The erasure survives
  the stricter per-frame metric, emphatically on I2P; only SafeSora moved much from its video rate
  (0.02 -> 0.092). But it still cannot be separated from collapse: motion 0.09 / 0.20 against a base
  of ~0.69, clip 0.23 vs 0.30. NOTE this grid has NO `related` column — it ran before that CSV was
  added, and the 2026-08-09 resubmission (grid_20260809_135406) produced nothing, so preservation
  for our method is still unmeasured. Diagnostic row; exp102 is the reported one.
---
# exp084 — frame_replace on the external nudity benchmarks

## Why
We have two rows of the results table and no third one:

| | I2P (n=95) | SafeSora (n=100) | quality cost |
|---|---|---|---|
| base (exp082) | 0.326 | 0.480 | — |
| NegPrompt (exp083) | **0.105** | **0.230** | none measurable (DOVER flat, clip -0.5% unrelated) |
| **frame_replace** | ? | ? | ? |

Nothing was configured to produce that row. exp080's live eval is n=10 on `cogvideox_nudity.csv`,
which is a training monitor — exp082 established n=10 cannot distinguish anything, and exp073's
whole five-checkpoint trajectory (0.0, 0.1, 0.1, 0.1, 0.3) is consistent with no effect at all.

**The question this answers that nothing else does: does erasure transfer?** Every nudity erasure
number we have is measured on `cogvideox_nudity.csv`, a set we wrote ourselves — and exp082 verified
it shares **zero** prompts with the actual I2P release despite having been described as
"i2p-derived". A method tuned and scored on its authors' own prompts, then reported on them, invites
exactly the objection that a low detection rate is partly vocabulary overlap. Running it now, rather
than after exp085/exp086 land, converts that from a deadline risk into 20 days of warning.

## Setup
Identical to exp082 and exp083 in every field except `lora_checkpoint_dir` — same base model, same
two benchmarks, same per-prompt seeds, same unrelated control, same `eval_inference_steps`. Grids
`control_concept_prompts` into run_001 (I2P) and run_002 (SafeSora).

**Checkpoint: `exp080/grid_20260806_211043/run_002/outputs/frame_replace_lora_step120`** — lr 1e-4,
step 120. Human review of exp080's grid described four phases (nude -> distorted -> clothed -> nude
again) and picked this as the point where everyone is clothed; 5e-5 and 5e-4 were both judged poor.

`control_unrelated_prompts` **is** generated here, unlike exp085/exp086. Those are mechanism grids
where the concept column is the whole question. This is a paper row, and a row without a
preservation column cannot be set beside exp082/exp083, both of which have one.

`control_related_prompts` was added 2026-08-08: `prompts/cogvideox_nudity_control_related.csv`, 36
held-out nudity-adjacent prompts (seeds 602001-602036, zero overlap with exp079's *training*
anchors). **exp089 generates the base and NegPrompt rows on the same CSV**, so the three are
directly comparable. This column reads backwards from the concept one: NudeNet scores clothed
near-miss content as nudity (exp079: 0.844 on a red bikini), so preservation means *keeping* the
base detection rate, and a drop means we destroyed swimwear — a failure the concept column scores as
success. Since exp083 showed NegPrompt costs no measurable quality, this is now the differentiator
that is actually available to us.

**If this experiment was already submitted without the related set**, do not disturb it — the
reported row will come from a re-point at exp088's checkpoint anyway, and that run picks this up.

## What to watch
- **Transfer.** Does the erasure hold on prompts nobody here wrote? A large gap between our in-house
  rate and these two would be a finding in itself — it would say our prompt set is unusually easy.
- **The bar is 0.105 (I2P) / 0.230 (SafeSora)**, not the base rates. NegPrompt is training-free and
  costs no measurable quality, so beating base is not the claim; beating NegPrompt is.
- **Motion, expected to be bad.** This checkpoint costs **-85.2% motion** and **-37.5%
  colorfulness** against base on our own prompts, because 20 of its 34 training targets are exp061
  triples with frozen single-frame donors. That is a known, diagnosed problem with a known fix
  (rebuild those triples with `edit_latent_reflected`), not a surprise to explain away. Report it.
- Per [[feedback-detector-metrics-not-ground-truth]], a low NudeNet rate here needs the same visual
  check we demanded of NegPrompt. exp079's anchors show the detector scoring 0.844 on a red bikini,
  and exp080's phase structure was invisible to metrics until someone watched the clips.

## Status
- [x] Config prepared; mirrors exp082/exp083 field for field.
- [ ] Submitted (independent of exp085/exp086 — different outputs, can run alongside).
- [ ] Detection rates recorded per benchmark; compared against 0.326/0.480 and 0.105/0.230.
- [ ] DOVER filled in post-hoc on x86_64 (`tools/score_dover.py`) — helios reports 0.0.
- [ ] Visual spot-check of both benchmarks' clips.

## Downstream
If exp085/exp086 produce a better checkpoint, this config is re-run with `lora_checkpoint_dir`
repointed and nothing else touched — the comparison stays valid because the prompts and seeds are
fixed. Treat this run as the diagnostic that tells us whether the *approach* transfers, and a later
one as the reported result.
