---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  CAPACITY IS THE FIRST LEVER THAT GENERALIZES, AND IT MOVES ALL FOUR CELLS AT ONCE. Restricted
  (10-way) row: ESR-1 67.86, ESR-5 20.92, PSR-1 85.28, PSR-5 93.92 — every one better than exp134's
  rank-8/eta-2.0/25-row baseline (49.90 / 15.61 / 82.71 / 93.19), and unlike eta (exp137) or dataset
  size (exp140), which each bought at most noise on ESR-5 while eta cost preservation, rank 32
  improves ESR-1 (+17.96), ESR-5 (+5.31) AND both PSR cells simultaneously — no trade-off. This is
  also the THIRD instance of a live 9-prompt monitor showing "top-5 hits 0.00", after exp135 and
  exp139 were both falsified by their full-protocol follow-ups (exp137, exp140) — this time it did
  NOT get falsified: chain saw's own restricted top-5 dropped to 0.79 (vs base ~1.0), below every
  prior rank-8 run's ~0.84-0.85 floor. exp142's second, independent worry — live-sample concept
  motion collapsing to 0.061, under GOAL.md's 0.15 guard floor — also did NOT generalize: full-protocol
  chain-saw motion_score_mean is 0.223, comfortably above floor (though the lowest margin of any
  rank-8/eta/dataset arm: exp134 0.390, exp137 0.371, exp140 0.262). Still short of GOAL.md's target
  (ESR-1 92.38, ESR-5 77.09) by a wide margin, especially ESR-5 (gap 56.17 points) — capacity helps
  but has not closed it. Next: push the lever further (rank 64, same lr/step budget) to see whether
  the ESR-5 gain continues or plateaus — exp147.
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

## Results (2026-08-24) — capacity is a real lever; both pre-registered risks did not generalize

Completed on helios, job 21074164, 2.53h of a 14h budget.

Restricted (10-way) row, against the three closed-lever baselines:

| run | ESR-1↑ | ESR-5↑ | PSR-1↑ | PSR-5↑ | erased-class motion |
|---|---|---|---|---|---|
| exp134 (rank 8, eta 2.0, 25-row) | 49.90 | 15.61 | 82.71 | 93.19 | 0.390 |
| exp137 (rank 8, eta 3.0) | 53.57 | 10.31 | 81.03 | 91.55 | 0.371 |
| exp140 (rank 8, eta 2.0, 47-row) | 52.55 | 15.82 | 81.34 | — | 0.262 |
| **exp143 (rank 32, eta 2.0, 47-row)** | **67.86** | **20.92** | **85.28** | **93.92** | **0.223** |

**Hypothesis A (capacity) holds, and cleanly.** ESR-5 moved +5.31 over the rank-8 baseline —
larger than either eta (exp137: -5.30, i.e. worse) or dataset size (exp140: +0.21, noise) managed —
and ESR-1 moved +17.96, the biggest single-lever jump in the thread. Unlike eta, which bought its
ESR-1 gain by giving up PSR-1/PSR-5, rank 32 improves *both* preservation cells too (PSR-1 85.28 vs
82.71, PSR-5 93.92 vs 93.19). Chain saw's own restricted top-5 is 0.7908, the first rank-8-beating
read of the residual-signal metric that every lever before this one left stuck at ~0.84-0.85 — this
is the concrete number behind the ESR-5 gain, not just a top-1 wobble redistributing rank.

This is also the third time a live 9-prompt monitor showed "concept top-5 hits 0.00" (after exp135
and exp139), and the first time that signal was NOT falsified by the full protocol — it correctly
predicted a real, if partial, improvement. The base rate on this specific live-monitor signal is
now 1/3 predictive; still worth reading with skepticism, but no longer purely a false-positive
generator.

**Hypothesis B (motion) is falsified.** exp142's live sample read concept motion collapsing to
0.061 by steps 500-600, under the 0.15 guard floor. The full-protocol number is 0.223 — comfortably
above floor, though the thinnest margin of the four rows above (0.223 vs floor 0.15, a 49% margin,
against exp134's 160% margin). The live sample's motion pessimism did not generalize, mirroring how
its top-5 optimism partially did — two more data points for "the 9-prompt live monitor is not the
protocol" in either direction, per exp071/exp133/exp135/exp139's standing lesson.

**Still short of GOAL.md's target.** ESR-1 67.86 vs threshold 92.38 (gap 24.52); ESR-5 20.92 vs
guard 77.09 (gap 56.17, the binding constraint). PSR-1 85.28 and PSR-5 93.92 both clear their
floors (54.03, 82.14) with the widest margin any arm in this thread has shown. Capacity is a real,
non-null lever — the first one found — but on this single doubling it closes at most a third of the
ESR-1 gap and a fraction of the ESR-5 gap.

## Status
- [x] Submitted (helios job 21074164, completed 2026-08-24T23:34).
- [x] Row measured under both conventions; checked against GOAL.md's target table and all four
      guards. Motion guard passes (0.223 vs 0.15 floor); ESR-1/ESR-5 both still below target/guard.
- [x] Compared against exp134/exp137/exp140: capacity is a real lever, not a third null result —
      it is the only one of the three so far that moves ESR-5 without costing PSR. Next: exp147
      pushes rank to 64 at the same lr/step budget to see whether the gain continues or plateaus.
