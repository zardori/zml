---
status: ready
concept: nudity
method: eval
thread: nudity
takeaway: >
  Pre-staged full battery (exp112's exact coverage) for whichever candidate survives DOVER + human
  review. `lora_checkpoint_dir` is the single field to fill; candidate paths and their n=25 numbers
  are in the config header. Blocked only on the DOVER results for exp123 r1 s80 / r2 s140 /
  exp136 r1 s200. 4 jobs.
---
# exp144 — full battery on the final candidate

## Why pre-staged
Every remaining candidate needs identical treatment, and n=25 cannot produce a reportable number
(it produced two false "0.0000" checkpoints already). Staging now means the battery submits the
same day the candidate is chosen.

## Selection rule
1. **DOVER-t >= 0.058** (>=83% of base 0.070) — the sharp band; everything at eta >= 4 fails this
   and matches the "not sharp" review verdict.
2. **Two adjacent checkpoints** at the claimed rate.
3. **Human review** decides between survivors — it has overturned the metrics three times in this
   thread (exp105's erasure, exp124's distortion, the colorfulness proxy itself).

## After it runs
The row drops into `docs/comparability_t2vunlearning.md` §4 beside base / NegPrompt / exp080 /
exp110, and if it wins, [exp113](../exp113_vbench_utility_gen4/notes.md)'s utility pair should be
re-run on the same checkpoint (one field) so the utility table matches the erasure table.

## Status
- [ ] Candidate chosen (blocked on DOVER for three checkpoints + human review).
- [ ] `lora_checkpoint_dir` filled; submitted (4 jobs).
- [ ] Row added to the comparability doc.
- [ ] exp113 utility re-run on the same checkpoint.
