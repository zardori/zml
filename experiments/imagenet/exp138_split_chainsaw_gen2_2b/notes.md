---
status: done
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  NOT falsified, though the yield sits a bit below exp131's. `tools/screen_split_dataset.py` on the
  full 30-row build: 22/30 pass (73%), 0 not-split, 8 no-concept, 0 blank-target — against exp131's
  25/30 (83%) on the same csv/recipe/model with fresh seeds only. A 10-point drop, all of it in
  `no-concept` (prompt/framing misses, same failure category as every prior build) and zero
  `not-split` failures (trajectory mode's fix still holds on these seeds too), so this lands inside
  the falsification band's "not seed luck" reading, just at the lower end of it — closer to a
  confirmation than exp131's own margin over exp121's 5b analogue (40% vs 47%). Survivors balance 12
  first / 10 second. Screened set written to `outputs_20260822_023755_screened.json`. Merged with
  exp131 (25 rows) the chain-saw 2B training set is now 47 rows, 24 first / 23 second — handed to
  exp139.
---
# exp138 — second-generation chain-saw split-prompt dataset on CogVideoX-2B

## Why
exp137 closed the `erase_esd_eta` avenue and did so against its own pre-registered falsification
criterion: eta=3.0's step-300 checkpoint, which looked like a top-5 win on exp135's 9-prompt live
monitor (top-5 hit 0.00 at steps 250/300, something no eta=2.0 checkpoint ever showed), lands at
restricted ESR-5 10.31 on the full 200-prompt protocol — *below* exp134's eta=2.0 baseline (15.61),
not above it. ESR-1 moved only 49.90 → 53.57, +3.7 points against a 92.38 target. Preservation cost
went the wrong way too: mean motion loss on the nine preserved classes (computed against exp130's
per-class base numbers) is ~36% at eta=3.0 vs exp134's ~32% at eta=2.0, and cassette player
specifically dropped to motion 0.039 (base 0.560, -93%) — inside 5b's frozen-poster range
(0.01-0.05) even though the erased-class motion guard (chain saw itself, 0.371) still clears its
0.15 floor comfortably. So more erase pressure is not the lever: it buys a few ESR-1 points at a
growing preservation cost while making the actual target metric worse. exp135's own write-up named
the alternative: "the next lever to try is dataset size or diversity, not this one."

exp131 is the entire dataset behind every 2B chain-saw run so far — 25/30 screened rows from
exp117's closeup prompts, trajectory mode. That is the same size the 5b thread had after exp117
alone, before exp121 (identical recipe, fresh seeds, `chain_saw_closeup_gen2.csv`) doubled it to 33
rows merging with exp066. This build repeats exactly that step for 2B: same csv, same recipe
exp131 already validated at 83% yield with 0 `not-split` failures, different seeds.

## Hypothesis and what would falsify it
Hypothesis: the gen2 seeds screen at a yield comparable to exp131's 83% (25/30), on the same
recipe/base model, confirming the yield is a property of the prompts and base model rather than
seed luck — the same check exp121 ran for 5b (12/30 = 40% against exp117's 14/30 = 47%, close
enough to confirm).

Falsified by:
- Yield far off 83% (e.g. below ~50% or nailing 100%) — would mean exp131's yield was seed luck,
  not a property of the prompt/recipe/model combination, and the merged dataset's composition
  should be treated with more suspicion.
- A `not-split` failure reappearing (trajectory mode's fix, confirmed on exp131 with 0/30, regressing
  on a new seed set) — would mean the suppression mechanism exp120/exp127 diagnosed on 5b is only
  partially fixed at 2B and re-open that question rather than treating it as settled.

This build alone does not test whether more data reduces the top-5 residual — that needs merging
with exp131's screened set, training, and a full eval, queued once this is screened.

## Setup
Field-for-field exp131 except `csv_path`: `chain_saw_closeup_gen2.csv` (exp121's gen2 seeds
3231-3260) in place of `chain_saw_closeup.csv` (exp117/exp131's original 30). Model
(`THUDM/CogVideoX-2b`), split geometry, `split_mode: trajectory`, concept/threshold all unchanged,
so any yield difference from exp131 is attributable to seed alone.

## What to watch
- **Screened yield** (`tools/screen_split_dataset.py`) against exp131's 25/30 (83%).
- **`not-split` / `no-concept` failure counts** against exp131's 0 / 5.
- **Positional balance** (first/second half) of survivors, to decide whether the merged set needs a
  rebalancing pass before training (exp121's 12 rows skewed 9/3, absorbed fine into the larger
  merge; exp138's own skew, if any, should be checked the same way before exp139 trains on it).

## Result

`tools/screen_split_dataset.py` on the full 30-row `metadata.json`:

```
30 clips | pass 22 (73%) | not-split 0 | no-concept 8 | blank-target 0
surviving concept_region balance: 12 first / 10 second
```

Against exp131's 25/30 (83%) on the identical prompts/recipe/model with only the seeds changed,
this is a real but modest drop, entirely inside the `no-concept` bucket (5 → 8) with `not-split`
still at zero — the trajectory-mode fix generalizes across seeds, and the loss is prompt/framing
sensitivity to the seed draw, not a regression in the splice mechanism. Falls short of "within noise
of exp131" but nowhere near the falsification band (below ~50%, or nailing 100%), so read as
confirmation with a wider margin than exp121's 5b analogue (12/30=40% vs exp117's 14/30=47%, a
7-point gap against this run's 10-point one) — both readings say yield is a property of the
prompt/recipe/model combination, with seed-to-seed variance on the order of 10 points either way.

## Status
- [x] Submitted (helios job 20944564, 2026-08-22).
- [x] Completed 2026-08-22T04:01 (exit 0, 37min of a 3h budget).
- [x] Screened, yield and failure breakdown checked against exp131: 22/30 (73%) vs 25/30 (83%), same
      failure category (no-concept only), not seed luck.
- [x] Merge with exp131 assembled (`extra_sources_file` on exp139's config, in-process at job start
      via the new `Config.extra_sources_file` field) and handed to exp139.
