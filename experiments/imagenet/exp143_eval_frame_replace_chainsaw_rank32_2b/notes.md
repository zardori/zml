---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
---
# exp143 — reported ESR/PSR for exp142's rank-32 chain-saw LoRA (2B), the capacity lever's full-protocol test

## Why
Two single-lever tests have already closed on this thread's target metric (restricted ESR-5):
`erase_esd_eta` above 2.0 (exp135's promising live signal falsified by exp137's full-protocol
ESR-5, 10.31 — *below* the eta=2.0 baseline of 15.61) and dataset size (exp139's promising live
signal falsified by exp140's full-protocol ESR-5, 15.82 — flat against baseline). exp142 is the
third lever, LoRA capacity (rank 8 → 32, at exp139's exact lr/step budget to remove exp141's
lr-scaling confound), and its live 9-prompt monitor shows the same shape that failed twice before:
top-1 converges cleanly (0.00 from step 200) and top-5 hits 0.00 at 4 of 6 checkpoints, below the
0.11-0.28 band every rank-8 run's live sample was stuck at.

Per exp142's own pre-registered gate ("full eval queued only if the live top-1 trajectory is
healthy"), that convergence earns the eval. But going in, the prior base rate (2/2 similar signals
falsified) argues for treating this as "the third test of the same question," not "a new lead" —
and exp142 surfaced a second, independent concern: its live sample's concept-class motion fell to
0.061 by steps 500-600, already below GOAL.md's 0.15 guard floor and lower than any rank-8 run's
final live reading. This run settles both at once: does capacity move ESR-5 where eta and dataset
size didn't, and does rank 32 cost the motion guard in a way eta/dataset-size increases did not.

## Hypothesis and what would falsify it
Hypothesis A (capacity): restricted ESR-5 exceeds exp134's rank-8/eta-2.0/25-row baseline of 15.61
by more than noise (the ~0.2-point moves exp137/exp140 showed count as noise; this needs a clearer
gap to count as a real lever).

Falsified by: ESR-5 landing within ~1-2 points of 15.61, or below it — the same "live top-5 hits
0.00, full protocol says nothing changed" pattern as exp135→exp137 and exp139→exp140. If so,
capacity joins eta and dataset size as a closed, null lever, and the thread has now exhausted the
three obvious single-variable knobs without closing GOAL.md's gap — worth a `needs_human` on
whether frame_replace's residual top-5 signal is a hyperparameter problem at all, versus something
structural about ResNet-50 top-5 neighborhoods for "chain saw" (chainsaw-adjacent tool classes) that
no amount of erasure pressure removes.

Hypothesis B (motion): the erased-class motion guard (0.15 floor) fails on this checkpoint, unlike
every eta/dataset-size run so far (exp134: 0.390, exp137: 0.371, exp140: 0.262 — all comfortably
above floor). exp142's own live sample already sits at 0.061, well below floor, on the identical
checkpoint under test.

Falsified by: erased-class motion landing above 0.15 on the full protocol — would mean the live
sample's motion collapse doesn't generalize the way its top-5 optimism might not either, i.e. two
opposite mistakes from the same 9-prompt sample.

## Setup
Field-for-field exp134/exp140 except `lora_checkpoint_dir` points at exp142's final checkpoint
(`experiments/imagenet/exp142_frame_replace_chainsaw_rank32_matched_lr_2b/outputs_20260822_224618/frame_replace_lora_step600`).
Same 200-prompt protocol, same `erased_class: "chain saw"`, same eval_inference_steps: 50.

## What to watch
- **Restricted ESR-1/ESR-5/PSR-1/PSR-5** against GOAL.md's target table and all four guards.
- **Erased-class motion** specifically — the guard this run is most likely to fail, per exp142's
  own live-sample warning. Compare against exp134 (0.390), exp137 (0.371), exp140 (0.262) to see
  whether rank 32 is a genuinely different (worse) regime for this metric, not just noisier.
- **Restricted top-5 on chain saw itself** — has stayed near 0.84 (barely below base 1.0) on every
  prior rank-8 run despite each one's live sample looking better; this is the number that decides
  whether capacity is a real lever or the third instance of the same optimism.

## Status
- [ ] Submitted.
- [ ] Row measured under both conventions; checked against GOAL.md's target table and all four
      guards, especially the motion floor.
- [ ] Compared against exp134/exp137/exp140 to settle whether capacity is a real lever or a third
      null result closing the last obvious single-variable knob.
