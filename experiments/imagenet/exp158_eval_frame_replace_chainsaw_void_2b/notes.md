---
status: done
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  SPLITS: THE VOID TARGET WINS ON THREE OF FOUR CELLS AND MOTION, BUT NOT ON ESR-5. Restricted
  (10-way) row: ESR-1 70.92, ESR-5 15.41, PSR-1 85.95, PSR-5 95.44. Against exp134's reported row
  for the identical rank-8/eta-2.0/600-step recipe on exp131's random-distractor dataset (ESR-1
  49.90, ESR-5 15.61, PSR-1 82.71, PSR-5 93.19): ESR-1 up sharply (+21.02), PSR-1 up (+3.24), PSR-5
  up (+2.25) -- all comfortably clear of GOAL.md's floors -- but ESR-5 is FLAT (-0.20, inside
  noise). So per the pre-registered falsifier this is a partial confirmation: prompt_b's content
  IS a real lever (HINTS.md's nudity-thread finding transfers to objects), but it moves ESR-1/PSR,
  not the residual-top5 problem every other single lever (eta: exp137, dataset size: exp140,
  capacity: exp143 only partially) has struggled with. Chain saw's own restricted top-5 confirms
  this directly: 0.8459 here vs exp134's 0.844 -- essentially unchanged, so the object still shows
  up in the model's top-5 guess about as often either way; the void target demotes it from #1 more
  reliably without dislodging it from the top 5. exp157's SECOND live-monitor read -- that concept
  motion would not collapse the way it did with every prior "clean top-5" candidate (exp135,
  exp139, exp142, exp147) -- DID generalize: erased-class motion_score_mean is 0.6897 against
  exp130's base of 0.8396 (-18%), the healthiest margin of any rank-8 checkpoint in the thread
  (exp134 -54%, exp137 -56%, exp140 -69%). Preserved-class mean motion loss (recomputed against
  exp130's per-class base) is ~34% -- essentially the same as exp134's ~32%, so the erased class's
  motion health does not extend to a preservation-motion win; two classes (tench +6.9%, golf ball
  +19.6%) even gain motion while cassette player (-53%), gas pump (-71%) and French horn (-60%)
  lose heavily, the same weak-class pattern (cassette player, gas pump) seen in exp137/exp140/
  exp148. Net: void target is now this thread's best lever for ESR-1 and both PSR cells
  simultaneously, and the first to improve the motion guard's margin rather than erode it, but it
  does not touch ESR-5, so it is a lever to STACK with the one lever that has moved ESR-5 (rank +
  early stopping: exp150's rank-32/step-300 restricted ESR-5 38.67), not a substitute for it.
  exp160 trains exp156's void dataset at rank 32 to test whether the two effects combine.
---
# exp158 — full esr_psr eval of exp157's void-target chain-saw LoRA, CogVideoX-2B

## Why
exp157 trained the thread's baseline rank-8/eta-2.0/600-step recipe on exp156's void-target
dataset (consistent "empty and bare" prompt_b instead of a different random distractor object per
row — HINTS.md's nudity-thread lever, ported to objects). Its live 9-prompt monitor is the
healthiest yet for a rank-8 run: concept top-1 is 0.00 at every checkpoint including step 100
(exp133's identical recipe on exp131's random-distractor dataset took until step 200), top-5 stays
low and noisy (0.00–0.10, versus exp133's reported 0.11–0.22), and — unlike every prior instance of
a clean live top-5 read (exp135, exp139, exp142, exp147) — concept motion does **not** collapse
alongside it (ends at 0.562, nowhere near the 0.15 guard floor, versus exp133's own final read of
0.140).

That combination (fast, low top-5 suppression *without* the motion collapse that co-occurred with
strong suppression in prior candidates) has not been seen before in this thread. But the thread's
own repeated lesson is that this exact class of live signal — top-5 reading near zero on 9 prompts
— has gone both ways on the full 200-prompt protocol: confirmed once (exp142→exp143, rank 32) and
nulled twice (exp135→exp137, eta; exp139→exp140, dataset size). This eval is the only way to know
which outcome exp157 gets.

## Hypothesis and what would falsify it
Hypothesis: the void-target dataset moves ESR-5 relative to exp134's reported row for the
identical rank-8/eta-2.0/600-step recipe on exp131's random-distractor dataset (restricted ESR-1
49.90, ESR-5 15.61, PSR-1 82.71, PSR-5 93.19), without giving back PSR below GOAL.md's floors
(54.03 / 82.14) or the motion guard (0.15).

Falsified by: ESR-1/ESR-5 landing within noise of exp134's row — would mean prompt_b's content
(void vs random distractor) is not a lever at this rank/eta/step budget, joining eta (exp137) and
dataset size (exp140) as single levers that looked promising live and were null on the full
protocol.

## Setup
Same shape as exp134: `mode: imagenet`, 200-video protocol, `erased_class: "chain saw"`.
`lora_checkpoint_dir` points at exp157's final checkpoint (step600) — chosen for the same reason
exp134 evaluated exp133's final checkpoint: top-1 never wavers from step 100 onward, so any earlier
checkpoint choice would be selection on the eval set with no live-monitor justification.

## What to watch
- Restricted (10-way) ESR-1/ESR-5/PSR-1/PSR-5 against exp134's row, cell for cell.
- Chain saw's own restricted top-5 (exp134's residual-signal problem: 0.842 vs base ~1.0) — does
  the void target move this specifically, or just top-1?
- Erased-class and preserved-class motion_score_mean against exp130's per-class base, per the
  guard and the thread's now-standard mean-preserved-motion-loss reading.

## Status
- [x] Submitted.
- [x] Compared against exp134's row.
- [x] Verdict written back into this file and exp157's.
