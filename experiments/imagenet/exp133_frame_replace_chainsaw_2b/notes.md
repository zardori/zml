---
status: done
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  RECIPE TRANSFERS. Concept top-1 on the 9-prompt live set follows exp069's 5b trajectory almost
  exactly: 0.09 -> 0.00 by step 200, holding at 0.00 (one blip of 0.01 at step 500) through step
  600 — erasure is real and stable at eta 2.0, same as 5b. Top-5 settles around 0.11-0.14, not
  0.00, so residual signal survives the way it did on 5b (exp071's restricted-convention ESR-5 was
  only 10.0 despite ESR-1 49.0). The live-set motion signature is DIFFERENT from 5b, and better:
  concept motion drops 0.339 -> 0.140 (step 100 -> step 600, -59%) but the 9-prompt "unrelated"
  sample actually RISES 0.349 -> 0.473 (+35%), the opposite of exp071's finding that the preserved
  classes lose a mean 45% of their motion on 5b. Per exp071's own lesson this is a 9-prompt live
  sample, not the protocol — it does not settle whether the freeze is concept-conditional at 2B;
  it only means 2B's live monitor does not show the global collapse 5b's did, which is worth
  checking against the full 200-prompt run rather than assuming it. Final checkpoint
  (`frame_replace_lora_step600`) is the one to report, same "no reason to deviate" logic as
  exp071 (erasure is flat at 0.00 from step 200 through 600, so choosing among checkpoints would
  be selection on the eval set). Unblocks exp134: the full ESR/PSR pass this pilot has been
  building toward.
---
# exp133 — frame_replace erasure of CHAIN SAW, on CogVideoX-2B

## Why
GOAL.md moves the whole object thread's base model to 2B so its ESR/PSR is a same-checkpoint
comparison against T2VUnlearning's Table 4 (2B-only for the objects protocol). Three gates already
cleared: exp130 (base 2B renders all ten protocol classes cleanly, restricted `Original`
ESR-1 10.60 / ESR-5 2.09 / PSR-1 89.40 / PSR-5 97.91, beating T2VUnlearning's published Original in
the baseline-appropriate direction), exp131 (trajectory-mode split-prompt recipe transfers to 2B at
83% pass, matching 5b almost exactly), exp132 (2B needs its own retention anchors because the VAE
`scaling_factor` differs — 1.15258426 vs 5b's 0.7 — and now has them: 30 entries, all ten classes).
This run is the actual training pass those three gates were clearing the way for: the first
chain-saw `frame_replace` LoRA on 2B.

## Hypothesis and what would falsify it
Hypothesis: the recipe that erased chain saw semantically on 5b (exp069: concept top-1 0.506 → 0.00
from step 200, scene intact) does the same on 2B, since exp131 already showed the split-prompt
mechanism it depends on is not 5b-specific.

Falsified by: concept top-1 failing to reach ~0.00 on the live eval set, or converging but the
scene collapsing outright (not just the known motion-freeze mode — a wholesale failure to
render the scene at all, which chain saw did not show on 5b). A result that erases nothing would
mean the split-prompt training target itself doesn't carry enough signal at 2B's smaller capacity,
which exp131's screening (a detector-confidence check on static frames) would not have caught.

## Setup
Field-for-field exp069 (5b chain-saw pilot) except:
- `model_id: THUDM/CogVideoX-2b`
- `metadata_file`/`latents_dir`: exp131's screened split-prompt set (25 rows, 83% pass, 12
  first / 13 second) in place of exp069's exp066+exp117 merge — a single source, so no
  `merge_dataset.sh` step is needed (same pattern exp070 used for exp118 alone).
- `retention_metadata_file`/`retention_latents_dir`: exp132's 2B preservation anchors in place of
  exp068's 5b ones — forced by the VAE scaling-factor mismatch exp131 found
  (`zml/unlearn/unlearn_frame_replace.py:323-326,346-349` hard-asserts it), not a free choice.

Everything else — `erase_esd_eta: 2` (exp126's only stable arm on 5b), lora rank/alpha, learning
rate, 600 steps / save_interval 100, timestep range, eval cadence, eval prompt CSVs — is unchanged,
so a difference in outcome is attributable to the base model and its forced dataset/retention swap,
not a hyperparameter confound exp126 already investigated separately.

## What to watch
- **Concept top-1** on the live eval set (`prompts/imagenet_objects/chain_saw.csv`, 9 sampled prompts
  per eval) — should trend toward 0.00 by step 200-300 if the 5b pattern holds.
- **Motion score on the erased class** — exp069/exp126 found the freeze (motion 0.010-0.049 against
  base 0.564) is not concept-conditional (exp071: the nine preserved classes lose it too) and not
  fixed by eta. GOAL.md's guard floor is 0.15; expect this run to test that floor for the first time
  on 2B rather than assume 5b's freeze severity carries over.
- **Preserved-class live eval** (`others_chain_saw.csv`) — sanity only; the real PSR number needs
  the full 200-prompt `esr_psr` eval (exp134-to-be), same as exp069→exp071.
- **Whether erasure is stable or oscillates** — exp126's eta=2.0 arm held 0.00 at all six
  checkpoints on 5b; if 2B oscillates instead (the failure exp070/exp128 saw on church, and exp126
  saw at eta 1.0/1.5), that would be a 2B-specific finding worth a note even though eta itself is
  unchanged here.

## Downstream
A checkpoint that erases (even with the freeze) is the input to the 2B counterpart of exp071: a
full 200-prompt `esr_psr` eval, which is the number this whole sub-thread has been building toward
— it is what gets compared against GOAL.md's target table. That eval is exp134.

## Results (2026-08-20) — recipe transfers, live-set motion signature differs from 5b

Completed on helios in 3.7 h (600/600 steps, job 20923171).

| step | concept top-1 | concept top-5 | concept motion | unrelated motion |
|---|---|---|---|---|
| 100 | 0.09 | 0.28 | 0.339 | 0.349 |
| 200 | 0.00 | 0.11 | 0.208 | 0.505 |
| 300 | 0.00 | 0.22 | 0.222 | 0.378 |
| 400 | 0.00 | 0.11 | 0.171 | 0.432 |
| 500 | 0.01 | 0.14 | 0.200 | 0.502 |
| 600 | 0.00 | 0.11 | 0.140 | 0.473 |

Concept top-1 lands at 0.00 by step 200 and holds (one 0.01 blip at step 500) — the same trajectory
and same stabilization step as exp069 on 5b. Top-5 does not reach 0.00 (settles 0.11-0.22), matching
exp071's finding that residual signal survives 5b's erasure too under top-5.

The motion story is not a repeat of 5b's: concept motion falls -59% (0.339 -> 0.140, step 100 to
600) while the 9-prompt "unrelated" sample *rises* 35% (0.349 -> 0.473) instead of collapsing with
it. On 5b (exp071, full 200-prompt protocol) the nine preserved classes lost a mean 45% of their
motion despite an even smaller live sample suggesting otherwise — so this run's live signal is not
proof the freeze is concept-conditional at 2B, only that the same small-sample optimism exp069 had
(later overturned by exp071) is present here too, in the opposite direction of what would be
worrying. Only the full 200-prompt run resolves it.

## Status
- [x] Submitted (helios job 20923171, 2026-08-20T18:18).
- [x] Completed 2026-08-20T22:02 (exit 0, 3.7h of a 12h budget).
- [x] Concept top-1 checked against exp069's 5b trajectory (0.506 → 0.00 by step ~200): matches —
      2B goes 0.09 → 0.00 by step 200, holds to step 600.
- [ ] Motion score checked against GOAL.md's 0.15 guard floor and the full-protocol preserved-class
      mean — needs exp134 (9-prompt live motion is not the protocol, per exp071's lesson).
