---
status: ready
concept: imagenet
method: frame_replace_split/precompute
thread: imagenet
takeaway: >
  Not yet run.
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
- [ ] Submitted.
- [ ] Screened, yield and failure breakdown checked against exp131.
- [ ] Decision: if yield is usable, queue a merge/train/eval cycle next tick to test whether the
      void target actually moves ESR-5 relative to exp153's rank-64/step-100 baseline (the thing
      this build cannot answer by itself).
