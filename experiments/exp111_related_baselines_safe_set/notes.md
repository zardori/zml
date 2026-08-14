---
status: done
concept: nudity
method: eval
thread: nudity
takeaway: >
  Base + NegPrompt on `prompts/ring_a_bell_nudity_safe.csv` (79 safe rewrites at matched seeds).
  exp102 measured OUR checkpoint at 0.0000 there, but with no baselines it is a cell rather than a
  column and cannot be reported. T2VUnlearning publishes no nudity preservation column at all, so
  this is a place to be strictly more complete than the comparison. 2 jobs.
---
# exp111 — baselines on the safe (related) set

## Why

`prompts/ring_a_bell_nudity_safe.csv` is 79 safe rewrites of the Ring-A-Bell nudity prompts at
**matched seeds** — same scene, same seed, clothed subject. It is the cleanest preservation probe we
have, because the only thing that differs from the erased distribution is the concept itself.

[exp102](../exp102_eval_frame_replace_comparable_nudity/notes.md) scored our checkpoint on it and got
**0.0000**. That number is currently unreportable: base and NegPrompt were never run on the set, so
there is nothing to compare against and no way to know what the detector's floor is on these prompts.
Given that base reads 0.043-0.053 on VBench people-prompts
([exp106](../exp106_vbench_utility_base/notes.md)), the floor here is probably non-trivial — and if
base reads, say, 0.05 on the safe set, then our 0.0000 is *below* the floor and that is a stronger
statement than "we score zero".

**T2VUnlearning reports no nudity preservation column of any kind** — only VBench utility, and only
on HunyuanVideo. So this is not catching up; it is a column we can have and they cannot.

## Design
Two arms in one config so prompts, seeds and inference settings are provably identical:

- `run_001` — empty negative prompt, i.e. the base model. Establishes the detector floor.
- `run_002` — the exact NegPrompt string used in exp083 / exp101 / exp102, so this row stacks with
  every other NegPrompt number in the comparable table.

The set is filed under `control_concept_prompts` because that is the slot scored by the concept
detector; it appears as **related** in the results table. **A low number here is the good outcome** —
these prompts contain no nudity, so this measures whether a method destroys adjacent-but-safe content.

## What to watch
- **The base row is the point.** It is the floor against which our 0.0000 and NegPrompt's number are
  read. Without it neither is interpretable.
- **Motion and DOVER on this set matter as much as the rate.** exp102's related motion reads 0.04 for
  our checkpoint; if base reads ~0.7 there, the freezing extends to safe adjacent content and that is
  a preservation failure the rate column alone would hide.
- DOVER will be 0.0 from helios (aarch64) — score locally with `tools/score_dover.py`.

## Results (2026-08-12)

`prompts/ring_a_bell_nudity_safe.csv`, 79 safe rewrites at matched seeds. **A low rate is good here**
— these prompts contain no nudity.

| | rate | motion | colour | clip |
|---|---|---|---|---|
| base (run_001) | 0.130 | 0.37 | 31.2 | 0.29 |
| NegPrompt (run_002) | 0.050 | 1.12 | 38.8 | 0.29 |
| **ours** (exp102, exp080 r2 s120) | **0.0000** | **0.04** | 28.2 | 0.27 |

### The good half
Base reads **0.130** on this set — a substantial detector floor on safe, clothed rewrites, higher than
the 0.043-0.053 exp106 measured on VBench people-prompts. Our 0.0000 is therefore not merely "zero",
it is **below the floor the base model itself produces on nudity-free content**, and below NegPrompt's
0.050. That is the strongest form this claim can take, and it is only sayable because the base row
exists.

### The bad half, which is the actual finding
**Motion falls 0.37 -> 0.04, a 89% loss, on prompts with no nudity in them.** This was flagged as the
risk to watch, and it is real: the freezing extends to semantically adjacent safe content.

Placed next to [exp107](../exp107_vbench_utility_frame_replace/notes.md), the damage is **monotonic in
semantic distance from the erased concept**:

| set | base motion | ours | delta |
|---|---|---|---|
| concept (Gen nudity) | 0.69 | 0.05 | **-93%** |
| **related (safe rewrites, matched seeds)** | **0.37** | **0.04** | **-89%** |
| VBench `object_class` | 0.92 | 0.29 | -68% |
| VBench `subject_consistency` | 1.60 | 1.03 | -36% |
| fire unrelated (15 prompts) | 2.01 | 1.62 | -19% |

So the edit is **graded, not confined**. "Localized erasure" is defensible for colour — exp107 shows
colorfulness is preserved off-concept — but not for motion, which decays smoothly with proximity and is
nearly as bad on safe rewrites as on the concept itself. This is the honest form of the preservation
story and it should be reported as this gradient rather than as a single unrelated-set number, which
(at -19%) flatters the method by a factor of nearly five.

## Status
- [x] Submitted and complete (2 jobs).
- [x] Base floor recorded (**0.130**); our 0.0000 and NegPrompt's 0.050 read against it.
- [x] Motion compared against exp102's related row — **-89%, the gradient finding above**.
- [ ] DOVER scored locally (helios wrote 0.0) — needs videos pulled; low priority, motion is the story.
- [ ] Re-run against exp110's checkpoint once chosen, since the whole table shifts.
