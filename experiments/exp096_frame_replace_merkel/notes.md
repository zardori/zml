---
status: ready
concept: face
method: frame_replace
thread: face_identity
takeaway: >
  frame_replace erasure of Angela Merkel, same regime as exp095 (Obama) with target_variant
  narrowed to exp095's winner. Blocked on exp093/exp094 and on exp095 picking a winner. Not yet
  submitted.
---
# exp096 — frame_replace erasure of Angela Merkel

## Why
Second pilot identity, completing the 2-identity pilot (`docs/comparison_targets.md` §2.3's "no grid
before the method is proven" rule — exp090's base row already covers all 5, so the pilot only needs
two erased identities to test whether frame_replace transfers to faces at all, not five). Confirms
whether exp095's result (erasure works / doesn't, which target_variant wins) generalizes across a
demographically different identity, or was specific to Obama.

## Setup
Field-for-field identical to exp095 except the dataset, `concept_target`, and the retention
exclusion — with one deliberate simplification: **`target_variant` is fixed to exp095's winner**,
not re-gridded. LR and target_variant are not variables under test in this run; running the full
`[split, wholeclip]` grid again would spend twice the compute to re-answer a question exp095 already
answers on the first identity. The config field below is a placeholder — **do not submit until
exp095 has picked a winner and this is updated to match.**

- Dataset: exp093 (split-prompt + whole-clip manufactured targets for Merkel).
- Retention: exp094's anchors minus Merkel's own (`retention_exclude`).

**Before submitting**, replace the `outputs_TIMESTAMP` placeholders with the real
`outputs_{timestamp}` directories from exp093 and exp094, and set `target_variant` to exp095's
actual winner.

## What to watch
Same as exp095 — read `summary.json` first, watch erasure vs. the shortcut test, `face_present_rate`
on both erased and preserved identities every checkpoint, and (if `target_variant: wholeclip` won)
motion collapse on the preserved set per R5.

**Specifically watch for identity-dependent results.** If Obama erased cleanly but Merkel doesn't (or
vice versa), that is itself the finding — worth understanding rather than averaging away, since it
would mean the split-prompt A/B/C recipe (or the erase regime) is not actually identity-agnostic
despite both datasets following the same construction.

## Downstream
exp098 runs the full 150-video ID-Similarity eval on the resulting checkpoint. Together with
exp095/exp097, this completes the pilot table: Original / NegPrompt / Ours for both identities,
comparable to T2VUnlearning's CogVideoX-5B Table 3 block.

## Status
- [ ] exp095 has a winning `target_variant`; this config updated to match.
- [ ] exp093 and exp094 complete; timestamps filled in.
- [ ] Submitted.
- [ ] Compared against exp095's result for identity-dependence.
