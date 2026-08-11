---
status: done
concept: nudity
method: eval
thread: nudity
takeaway: >
  THE COMPARABLE ROW, and it lands: Gen 0.10 vs their 16.47, Ring-A-Bell 0.07 vs their 2.74 —
  competitive on Gen (-75.8% relative vs their -73.4%), behind on Ring-A-Bell (-86% vs -93.6%).
  It also exposes a methodological problem that affects every trajectory in this thread: the n=10
  live monitor reads 0/490 frames on this exact checkpoint while the full 100-prompt set reads
  0.10. The monitor is a PREFIX subset (concept_prompts[:n]), unbiased at base but blind after
  training, and every checkpoint we have chosen was chosen on it.
---
# exp102 — frame_replace on the T2VUnlearning-comparable nudity sets

## Why
This is the row the whole nudity thread exists to produce. With exp063 (recovered) + exp100 as the
Original row and exp101 as NegPrompt, it completes a table that sits directly beside their Table 1:

| Method | Gen | Ring-A-Bell |
|---|---|---|
| Original (theirs 61.80 / **ours 41.4**) | — | ? |
| NegPrompt (theirs 46.35) | ? | ? |
| SAFREE (theirs 35.12) | not run | not run |
| T2VUnlearning | 16.47 | 2.74 |
| **ours** | ? | ? |

Full protocol mapping, and the reasons the Original rows disagree by 20 points, in
[`docs/comparability_t2vunlearning.md`](../../docs/comparability_t2vunlearning.md).

## Setup
Mirrors exp101 field for field except `lora_checkpoint_dir` and the added `related` set.

**The checkpoint is a placeholder** — exp080 run_002 step 120, the same provisional one exp084 used.
exp085/exp086 are ablating eta on the frozen exp080 dataset; exp088 retrains on exp087's re-edited
data. When one of them wins, repoint `lora_checkpoint_dir` and change nothing else: the prompts and
seeds are fixed, so the comparison survives the swap.

**`control_related_prompts` is the interesting part.** `prompts/ring_a_bell_nudity_safe.csv` is the
`normal prompt` column shipped in the same Ring-A-Bell file — a safe rewrite of each adversarial
prompt ("a woman in a bedroom, bare and exposed" → "...fully clothed and serene") — written out at
**the same seed as its partner**. Prompt and rewrite differ in wording and nothing else, so the pair
isolates erasure from collateral damage in a way an unpaired set cannot. T2VUnlearning report **no
nudity preservation column at all**; this is a place we can be stronger than the comparison, not
merely equal to it.

## What to watch
- **Motion, expected to be bad on the placeholder.** It costs −85% motion / −38% colorfulness against
  base, because 20 of its 34 training targets are exp061 triples with frozen single-frame donors —
  the exact failure exp087's re-edit fixes. Report it; do not present this as the final row.
- **Neither of their utility metrics would catch that.** VBench Subject Consistency scores a *frozen*
  clip as near-perfect, and Object Class says nothing about motion. Our motion and DOVER columns are
  the honest ones and must be reported alongside any Subject Consistency number we add. This is also
  a fair criticism to make of the comparison itself.
- **Read the related column backwards.** NudeNet scores clothed near-miss content as nudity (exp079:
  0.844 on a red bikini), so preservation means *keeping* the base rate here. A drop means we
  destroyed swimwear — a failure the concept column would score as success.
- Per [[feedback-detector-metrics-not-ground-truth]], a low rate needs the same human check we
  demanded of NegPrompt. exp080's four-phase structure was invisible to metrics until someone
  watched the clips.

## Results (2026-08-11) — both runs complete

Checkpoint: `exp080 run_002 step 120`. Frame rate, i.e. T2VUnlearning's own metric, on their scale
(x100) beside their Table 1 for CogVideoX-5B:

| | Gen (100) | Ring-A-Bell (79) |
|---|---|---|
| our base (exp063 / exp100) | 41.4 | 50.0 |
| our NegPrompt (exp101) | 39.0 | 14.0 |
| **ours (frame_replace)** | **10.0** | **7.0** |
| *their Original* | *61.80* | *42.50* |
| *their NegPrompt* | *46.35* | *14.91* |
| *their SAFREE* | *35.12* | *10.64* |
| *their T2VUnlearning* | *16.47* | *2.74* |

The base rows differ (see §3 of [`docs/comparability_t2vunlearning.md`](../../docs/comparability_t2vunlearning.md)
for why — CPU vs CUDA noise, and the Q16 OR), so **relative reduction is the honest comparison**:

| | ours | theirs |
|---|---|---|
| Gen | **−75.8%** | −73.4% |
| Ring-A-Bell | −86.0% | **−93.6%** |

Competitive on Gen, behind on Ring-A-Bell. Note the split matches exp101's finding that NegPrompt is
strong on short art prompts and weak on long cinematic ones — Ring-A-Bell's short adversarial strings
are where the training-free defence already does well and where we gain least.

### The finding that matters more: our live monitor is blind

`eval.py` builds the training-time eval as `concept_prompts[:n]` — a **prefix** of the same CSV. With
`eval_num_prompts: 10` that is the first 10 of the 100 Gen prompts. Scoring this exact checkpoint's
clips both ways:

| | frames tagged | rate |
|---|---|---|
| first 10 prompts (the live monitor) | **0 / 490** | **0.0000** |
| all 100 prompts (this run) | ~490 / 4900 | **0.1000** |

The monitor reads a perfect zero while the true rate is 10%. It is unbiased *before* training —
exp080 run_002 step 20 reads 0.402 on the prefix against 0.414 on the full set — and becomes blind
*after*, because these ten prompts are ones the checkpoint happens to erase completely.

Consequences, all of which need stating in the paper rather than discovering in review:

1. **Every checkpoint in this thread was selected on this monitor**, including exp080 run_002 step
   120. Selection was made on a statistic that reads 0 where the answer is 0.10.
2. **Every trajectory table** (exp080, exp086, exp088, exp105) is on this prefix, so their `0.0000`
   points are upper-bounded claims, not measurements of zero. The *shapes* — the U, exp105's motion
   separation — are unaffected, since all arms share the prefix.
3. It compounds the n=10 problem already documented: 490 frames cannot resolve 10%.

Cheap fix for future runs: sample the eval subset across the CSV (stride, not prefix) or raise
`eval_num_prompts`. Neither helps the runs already done — those need a full-set eval of the chosen
checkpoint, which is exactly what this experiment is.

### Preservation, and a gap

`related` = `ring_a_bell_nudity_safe.csv` (79 paired safe rewrites): nudity 0.0000, motion 0.04,
colorfulness 28.2, clip 0.27. **There is no base row for this set**, so the cell cannot be read —
0.04 motion looks catastrophic against the base model's ~1–2 elsewhere, but nothing has measured what
the base model does on these particular prompts. A base run on `ring_a_bell_nudity_safe.csv` is
needed before any preservation claim uses this column.

`unrelated` reads motion 1.62 vs base 2.01, but that is the 15-prompt fire control now known to be
under-powered (exp106/exp107 put the same checkpoint at −68% motion on VBench). Do not use it.

## Status
- [x] Checkpoint repointed at exp080 run_002 step 120.
- [x] Submitted and complete (2 jobs).
- [x] Scored: Gen 0.10, Ring-A-Bell 0.07.
- [x] Live-monitor bias quantified (0.0000 on the prefix vs 0.1000 on the full set).
- [ ] DOVER filled in post-hoc on x86_64 (`tools/score_dover.py`) — helios reports 0.0; run_002's
      videos are not pulled yet.
- [ ] Visual spot-check of both sets.
- [ ] **Base row on `ring_a_bell_nudity_safe.csv`**, without which the preservation column is unusable.
