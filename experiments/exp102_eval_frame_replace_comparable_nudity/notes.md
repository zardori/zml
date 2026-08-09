---
status: ready
concept: nudity
method: eval
thread: nudity
takeaway: >
  The comparable method row: frame_replace on T2VUnlearning's own Gen (100) and Ring-A-Bell (79)
  sets, plus the paired safe rewrites as a preservation column they do not report at all.
  lora_checkpoint_dir is a PLACEHOLDER pending exp085/exp086/exp088. 2 jobs. Not yet submitted.
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

## Status
- [ ] Checkpoint repointed at the exp085/exp086/exp088 winner (currently placeholder).
- [ ] Submitted (2 jobs; run_001 ~194 clips, run_002 ~173).
- [ ] DOVER filled in post-hoc on x86_64 (`tools/score_dover.py`) — helios reports 0.0.
- [ ] Scored on x86_64, same machine as every other reported row.
- [ ] Visual spot-check of both sets.
