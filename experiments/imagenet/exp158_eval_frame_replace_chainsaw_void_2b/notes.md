---
status: ready
concept: imagenet
method: eval
thread: imagenet
takeaway: >
  Not yet run.
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
- [ ] Submitted.
- [ ] Compared against exp134's row.
- [ ] Verdict written back into this file and exp157's.
