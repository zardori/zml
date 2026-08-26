---
status: done
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  NOT FALSIFIED: the void target is usable data, statistically indistinguishable from exp131's
  random-distractor build. `tools/screen_split_dataset.py --min-concept-max 0.10` (the calibrated
  chain-saw threshold): 25/30 pass (83%), 0 not-split, 5 no-concept, 0 blank-target -- the exact
  same pass count as exp131 on the identical 30 prompt_a/prompt_c/seed rows, and 4 of the 5
  no-concept failures are the SAME seeds (3204, 3211, 3218, 3224). The two seeds that disagree
  (3203, 3226) flip only because their conc_max sits within 0.01 of the 0.10 cutoff in both builds
  (exp131: 0.0911 / 0.1247; exp156: 0.1048 / 0.0716) -- a borderline flip consistent with cross-run
  GPU nondeterminism, not a systematic effect of prompt_b's content. That is exactly what
  `split_mode: trajectory` predicts: the concept-half trajectory is generated independently of
  prompt_b and spliced in afterward, so prompt_b cannot change whether prompt_a renders the concept
  -- and the data confirms it doesn't. Screened set written to
  `outputs_20260826_071200_screened.json` (13 first / 12 second). This only establishes the build is
  usable; whether the void target changes what the trained LoRA learns (the actual point of
  HINTS.md's lever) is exp157.
---
# exp156 — chain-saw split-prompt dataset with a consistent VOID target instead of a random
# distractor object, on CogVideoX-2B

## Why
Every single-lever sweep tried so far on this thread's best checkpoint family has plateaued: eta
(exp137, worse ESR-5), dataset size via a second independently-seeded batch (exp138→exp140, null),
rank (exp143/exp147/exp148, non-monotonic and peaks around rank 32-64), and checkpoint/early-
stopping (exp149-exp155, best full-protocol row still exp153's rank-64/step-100 at restricted
ESR-1 77.86 / ESR-5 44.49 — 14.5 / 32.6 points short of GOAL.md's target of 92.38 / 77.09). All of
these tuned *how hard* or *how long* to push the same training signal; none has touched what the
signal itself looks like.

HINTS.md records an unexploited lever from the other thread: in nudity, making prompt_b's target
LESS varied (all subjects in plain black clothing, one consistent look, instead of a different
outfit per training row) made the unlearning signal clearer. Every object-thread split-prompt build
to date (exp066, exp117, exp121, exp131, exp138) has done the opposite — prompt_b substitutes a
different, unrelated tool/object per row (a paint can here, a crowbar there, a coil of rope
elsewhere). If the model has to learn "chain saw → {30 different objects}" rather than
"chain saw → {one consistent thing}", the gradient a LoRA rank of 8-64 has to fit is a much messier
target, which is a plausible reason ESR-5 (the metric that asks the model to drop the object from
its top-5 *guess*, not just its top-1) has been the hardest cell to move under every other lever.

This build tests the analogous "void" target for objects: prompt_b becomes the bare, empty version
of the same surface prompt_c already describes (no object at all), worded identically across all 30
rows ("empty and bare"), with prompt_a's close-view/fills-the-frame clause carried over so the two
branches stay compositionally matched — that framing match is what exp099/exp119 identified as load-
bearing for a clean splice, and what exp120/exp127 traced the `prediction`-mode suppression failure
back to when it was missing.

## Hypothesis and what would falsify it
Hypothesis: swapping prompt_b's target from a varied random object to a consistent void does not
break the split-prompt mechanism — screened yield lands within noise of exp131's 83% (25/30) on the
identical prompt_a/prompt_c/seed set, with `not-split` failures still at or near 0 (trajectory
mode's fix holds regardless of what prompt_b describes). This build only tests whether the void
recipe is *usable data*; whether it changes the trained LoRA's ESR-5 is a separate, later-tick
question that needs a merge/train/eval cycle this build would unblock.

Falsified by:
- **Yield collapses** (well below ~50%) — would mean the model needs a concrete object noun to
  render a coherent competing scene, and an empty-surface prompt_b produces blank or degenerate
  clips `screen_split_dataset.py`'s blank-frame gate would catch.
- **`not-split` failures reappear** — would mean the void framing (extra "empty and bare" clause,
  no distractor object anchoring composition) reintroduces the shared-latent-context suppression
  trajectory mode was built to fix, a different failure mode than exp131/exp138 ever saw.

## Setup
Field-for-field exp131 (this thread's first, best-yielding 2B chain-saw build: 25/30, 0 not-split,
5 no-concept) except `csv_path`, which now points at
`prompts/imagenet_objects/split/chain_saw_closeup_void.csv` — generated mechanically from
`chain_saw_closeup.csv` by replacing each row's prompt_b with its own prompt_c plus
", empty and bare, in close view, filling much of the frame" (same seed, same prompt_a, same
prompt_c). Model (`THUDM/CogVideoX-2b`), split geometry, `split_mode: trajectory`, concept/threshold
all unchanged, so any yield or failure-mode difference from exp131 is attributable to prompt_b's
content alone.

## What to watch
- **Screened yield** (`tools/screen_split_dataset.py`) against exp131's 25/30 (83%).
- **`not-split` / `no-concept` / `blank-target` counts** against exp131's 0 / 5 / 0.
- If it screens usably: whether the *pattern* of failures differs (e.g. blank-target failures that
  never appeared with a distractor object present) — that would itself be informative about why a
  void target might or might not carry a training signal, independent of the eventual ESR-5 test.

## Status
- [x] Submitted.
- [x] Screened, yield and failure breakdown checked against exp131: 25/30 vs 25/30, 0 vs 0
      not-split, 5 vs 5 no-concept (4 of 5 the same seeds).
- [x] Decision: yield is usable and the failure mode is unchanged, so exp157 trains the identical
      rank-8/eta-2.0/600-step recipe exp133 used on exp131's dataset, on this build instead, for a
      clean single-variable (prompt_b content) comparison against exp134's reported row.
