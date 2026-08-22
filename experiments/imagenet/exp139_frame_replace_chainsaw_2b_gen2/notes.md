---
status: done
concept: imagenet
method: frame_replace
thread: imagenet
takeaway: >
  Live monitor is healthy and clears both falsification bars, so the run earns its full-eval
  follow-up (exp140). Concept top-1 reaches 0.00 by step 200 and holds through 600 (one 0.01 blip
  at step 500, the same pattern exp133 showed), so the extra 22 rows did not destabilize training.
  Top-5, the metric this run exists to move, hits 0.00 at steps 300 AND 600 -- below exp133's
  0.11-0.22 floor for its entire run. But exp135 showed this exact signal (live top-5 hitting 0.00)
  on the eta sweep and exp137's full 200-prompt eval then read WORSE than the eta=2.0 baseline it
  was supposed to beat, so this is a lead, not a result -- the full eval is the only thing that can
  confirm it. One point favors this run over exp135's: preserved-class motion here DROPS (unrelated
  0.384 -> 0.266, -31%) rather than rising as exp133's live sample misleadingly did (+35%, which
  exp134's full protocol then corrected to a ~32% loss) -- so this run's live sample is not
  repeating exp133's specific optimism failure, even if the top-5 number needs the same scrutiny
  exp137 gave eta. Final checkpoint (frame_replace_lora_step600, top-1 0.00 / top-5 0.00) is the one
  to eval, same "no reason to deviate" logic as exp071/exp134/exp137.

  CORRECTED 2026-08-22 by exp140: the live top-5-hits-0.00 signal did NOT survive the full
  200-prompt protocol. Restricted ESR-5 landed at 15.82, statistically flat against exp134's 15.61
  (rank-8/eta-2.0/25-row baseline), and chain saw's own restricted top-5 barely moved off base
  (0.842 vs 1.0). It joins exp069/exp133/exp135's small-sample optimism failures -- this run's
  healthy live monitor was not evidence the dataset-size lever works.
---
# exp139 — frame_replace erasure of CHAIN SAW on CogVideoX-2B, merged (exp131+exp138) dataset

## Why
exp137 closed the `erase_esd_eta` avenue by falsifying its own pre-registered hypothesis: eta=3.0's
step-300 checkpoint, which looked like a top-5 win on exp135's 9-prompt live monitor, lands at
restricted ESR-5 10.31 on the full 200-prompt protocol — *below* exp134's eta=2.0 baseline (15.61),
not above it. ESR-1 moved only 49.90 → 53.57 against the 92.38 target. Preservation cost also went
the wrong way (mean motion loss on the nine preserved classes ~36% vs eta=2.0's ~32%, cassette
player specifically dropping to 0.039, inside 5b's frozen-poster range). So more erase pressure is
not the lever. exp135/exp137's own write-up named the alternative: dataset size or diversity.

exp138 built that data: a second, independently-seeded 30-prompt batch on exp131's already-validated
closeup-prompt/trajectory-mode recipe, screening at 22/30 (73%) — a real but modest drop from
exp131's 25/30 (83%), entirely in the `no-concept` bucket (prompt/framing misses) with zero
`not-split` failures, so read as confirmation that the recipe transfers seed-to-seed, not as a
regression. Merged with exp131, the training set goes from 25 rows to 47 (24 first / 23 second,
still balanced).

## Hypothesis and what would falsify it
Hypothesis: nearly doubling the training set (25 → 47 rows), holding every other field at exp133's
established eta=2.0 baseline, improves the target metric — restricted ESR-5 — beyond exp134's 15.61,
without the preservation guards (PSR-1 ≥ 54.03, PSR-5 ≥ 82.14, erased-class motion ≥ 0.15) getting
worse than exp134's own 82.71 / 93.19 / 0.390.

Falsified by:
- Restricted ESR-5 at or below exp134's 15.61 on the full 200-prompt eval (needs a follow-up eval
  run once this training completes) — would mean dataset size/diversity is not the lever either,
  and the residual top-5 signal (chain saw staying in the model's top-5 guess even after top-1
  erasure) is a property of the recipe (eta=2.0, rank-8 LoRA, 600 steps) rather than of how much
  data it sees.
- The live 9-prompt monitor oscillating instead of holding top-1 at 0.00 from step 200 onward (the
  pattern every eta=2.0 run — exp069, exp133 — has shown so far) — would mean the added rows
  introduce noise the smaller set didn't have, worth flagging before spending a full eval on it.

This run alone reports only the live 9-prompt monitor, per exp071's own lesson that a small live
sample is not the protocol; a full `esr_psr` eval (exp139's own analogue of exp134) is the
follow-up this queues if the live signal looks healthy.

## Setup
Field-for-field exp133 (the 2B eta=2.0 baseline) except the training set: primary
`metadata_file`/`latents_dir` still point at exp131's 25 screened rows, and a new
`extra_sources_file` (`extra_sources.json`, this dir) names exp138's 22 screened rows as a second
source. Both are merged in-process at job start by the new `Config.extra_sources_file` field on
`zml/unlearn/unlearn_frame_replace.py` — see that file's code_changes entry for why this exists
instead of the usual `merge_dataset.sh` + ssh step (this agent role has no cluster access this
tick). `erase_esd_eta: 2`, lora rank/alpha, lr, steps, timestep range, eval cadence, retention set —
all unchanged from exp133, so any difference in outcome is attributable to the dataset, not a
confound with the eta sweep exp135/137 already ran separately.

## What to watch
- **Concept top-1** on the live eval set — should reach 0.00 by step ~200 and hold, matching
  exp133's trajectory, if the extra rows don't destabilize training.
- **Concept top-5** — the residual-signal metric this run exists to move. exp133 settled at
  0.11-0.22; any checkpoint clearly below that range on the live set is worth a full eval even
  before step 600.
- **Preserved-class motion** on the 9-prompt unrelated sample, as a cheap early warning — not the
  protocol number (that needs exp134's full-protocol read), but exp133's own live sample was
  optimistic in the wrong direction (unrelated motion *rose*), so this is a sanity check, not
  evidence either way on its own.
- **Positional balance** of the merged set (24 first / 23 second) — already checked, no rebalancing
  needed before this run.

## Status
- [x] Submitted. Completed on helios, 3.7h, exit 0 (job 20945839).
- [x] Live monitor checked against exp133's trajectory: top-1 0.00 from step 200 through 600 (one
      0.01 blip at step 500, matching exp133), top-5 hits 0.00 at steps 300 and 600 -- below
      exp133's 0.11-0.22 floor. Healthy by both pre-registered criteria.
- [x] Full `esr_psr` eval queued as exp140, using the final checkpoint
      (`frame_replace_lora_step600`). exp135/exp137 already showed a live top-5-hits-zero signal
      can fail the full protocol, so exp140's result is what actually decides this, not this
      write-up.
