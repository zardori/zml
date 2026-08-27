---
status: ready
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  Not yet run.
---
# exp173 — frame_replace: chain-saw void-target dataset x rank 16, CogVideoX-2B, fine-grained

## Why
Two ranks are now measured on exp156's void-target dataset, and both show the same qualitative
shape: a non-monotonic ESR-5 peak inside the first 300 steps, not a monotonic climb to either the
earliest or latest checkpoint.

- **Rank 8** (exp157, evaluated this tick at exp170/exp171/exp172, plus exp158's step-600):
  restricted ESR-1/ESR-5 = 64.80/23.98 (step 100) → 68.88/32.04 (step 200, peak) → 69.69/30.20
  (step 300) → 70.92/15.41 (step 600). Motion stays safe at every checkpoint (0.50–0.81 against
  the 0.15 floor).
- **Rank 32** (exp160, evaluated at exp161–exp169): restricted ESR-1/ESR-5 = 45.71/10.82 (step 600)
  → 60.41/18.47 (step 300) → 86.73/43.57 (step 200, peak) → 62.76/16.94 (step 100). Motion
  *breaches* the 0.15 floor exactly at the peak (0.1379), while the neighbouring checkpoints
  (150/175/225/250/300) all land back near step 300's level (ESR-5 15–18) and stay motion-safe.

So capacity appears to control the peak's amplitude and its motion risk together: rank 8 keeps the
peak small but legal, rank 32 makes it big but illegal at the exact step that matters. Rank 16 —
untried at any step budget — is the direct test of whether that trade is continuous (a legal peak
bigger than rank 8's exists in between) or a step function (any capacity increase either does
nothing or breaches the floor, with no usable middle ground).

## Hypothesis and what would falsify it
Hypothesis: rank 16 produces a step-~200 ESR-5 peak intermediate between rank 8's (32.04) and rank
32's (43.57), while its motion at that peak stays clear of the 0.15 floor (unlike rank 32's).

Falsified by: no checkpoint in the swept range showing a live-monitor classification signal
stronger than rank 8's own peak, OR the live motion signal at that peak dropping into rank 32's
failing range (~0.14) — either would mean the trade is not continuous, and rank 16 buys nothing
useful over rank 8 on this dataset.

## Setup
`job_type` defaults to `unlearn` (training). Field-for-field exp165 (same void-target dataset,
same 300-step/save_interval-25 fine-grained schedule, same lr/eta/preservation setup) except
`lora_rank`/`lora_alpha`: 16/16.0 instead of 32/32.0. Only the live 9-prompt monitor is read from
this run, per the thread's standing practice (exp155, exp165) — a full `esr_psr` eval is a
next-tick follow-up on whichever checkpoint the live trajectory names as the best candidate.

## Status
- [ ] Submitted.
- [ ] Live monitor reviewed, candidate checkpoint(s) selected for a full `esr_psr` eval.
