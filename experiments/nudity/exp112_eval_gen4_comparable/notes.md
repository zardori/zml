---
status: done
concept: nudity
method: eval
thread: nudity
takeaway: >
  OVERTURNED exp110's headline. On the FULL sets the gen4 checkpoint erases WORSE than the old
  incumbent everywhere (Gen 0.150 vs 0.100, Ring-A-Bell 0.250 vs 0.070, I2P 0.100 vs 0.005) while
  being much better on quality (Gen motion 0.14 vs 0.05, colour 33.4 vs 24.0). exp110's 0.0000 was
  an n=10 live-eval artifact and so was the old checkpoint's. Two operating points on a trade curve,
  not a winner. Also: `eval_num_prompts: 10` cannot rank checkpoints, which is how every checkpoint
  in this thread was chosen.
---
# exp112 — comparable table on the gen4 checkpoint

## Why
[exp110](../exp110_frame_replace_nudity_gen4/notes.md) step 140 reads rate 0.0000 at colorfulness
35.4 (base 36.3) and motion 0.25, against the old incumbent exp080 r2 s120's 0.0000 / 21.9 / 0.11.
Every number in [`docs/comparability_t2vunlearning.md`](../../docs/comparability_t2vunlearning.md)
is therefore measured on a superseded checkpoint.

This runs exp102's and exp084's coverage together so the table moves in one piece: Gen (100),
Ring-A-Bell (79), I2P (95), SafeSora (100), plus `related` and `unrelated`.

## The cell that matters most
Not the erasure rates — those are expected to hold, since the live eval already reads 0.0000 on 490
frames of the Gen set. It is **`related`**. [exp111](../exp111_related_baselines_safe_set/notes.md)
established base 0.130 / NegPrompt 0.050 there, and found the old checkpoint froze that set as
badly as the concept set itself (motion 0.37 -> 0.04, **-89%**, on prompts containing no nudity).
That gradient — -93% concept, -89% related, -68%/-36% VBench, -19% fire-unrelated — is the honest
form of the preservation story.

exp110 holds 2.3x the motion on concept prompts. If that carries to `related`, the gradient flattens
and "graded, not confined" softens into something much easier to defend. If it does not, the new
checkpoint is better on the concept set and no better where it matters for preservation.

## Checkpoint choice
**step 140**, and DOVER has now confirmed it (2026-08-14). It dominates step 120 on every axis:
identical rate (0.0000) and motion (0.25), 4 points more colorfulness, same clip score, and
**higher on both DOVER axes** (technical 0.0616 vs 0.0584, aesthetic 0.8871 vs 0.8413). The open
question when this was staged — whether 140's extra colour was artefacts rather than saturation —
is answered: it is saturation. No config change needed.

## Standing caveat
**Human review has not happened.** The clips are pulled and staged for it. Running the eval now is
fine — it costs hours and unblocks the table — but per [[feedback-detector-metrics-not-ground-truth]]
nothing here is reportable until the checkpoint has been watched. If review rejects step 140 this is
a one-field re-run.

## Status
- [ ] Submitted (4 jobs).
- [ ] Human review of exp110 step 140 (independent of this run, but gates reporting).
- [ ] DOVER scored locally on the outputs (helios writes 0.0).
- [ ] `docs/comparability_t2vunlearning.md` §4 rewritten on the new checkpoint.


## Results (2026-08-14)

Checkpoint: exp110 step 140. Baselines are exp063/exp100 (base), exp101/exp111 (NegPrompt),
exp102/exp084 (the old incumbent, exp080 r2 s120).

### Erasure — the gen4 checkpoint is worse on every set

| set | n | base | NegPrompt | old (exp080 r2 s120) | **new (exp110 s140)** |
|---|---|---|---|---|---|
| Gen | 100 | 0.414 | 0.390 | **0.100** | 0.150 |
| Ring-A-Bell | 79 | 0.500 | 0.140 | **0.070** | 0.250 |
| I2P | 95 | 0.346 | 0.137 | **0.0054** | 0.100 |
| SafeSora | 100 | 0.500 | 0.263 | **0.092** | 0.110 |
| related (safe) | 79 | 0.130 | 0.050 | **0.0000** | 0.020 |

Against T2VUnlearning on Gen: theirs -73.4%, old checkpoint **-75.8%**, gen4 checkpoint **-63.8%**.
So the *old* checkpoint is the competitive row and the new one is not.

### Quality — the gen4 checkpoint is better on every set

| set | motion old -> new | colour old -> new |
|---|---|---|
| Gen | 0.05 -> **0.14** | 24.0 -> **33.4** |
| Ring-A-Bell | 0.03 -> 0.04 | 24.0 -> **38.1** |
| I2P | 0.09 -> **0.15** | 32.5 -> **46.4** |
| SafeSora | 0.20 -> **0.37** | 24.7 -> **42.9** |
| related | 0.04 -> 0.06 | 28.2 -> **39.4** |

Base motion for reference: Gen 0.69, related 0.37. So the new checkpoint still loses 80% of the
motion on Gen — the improvement is from 93% to 80%, not to anything like preservation.

### The `related` question exp111 raised
exp111 found the old checkpoint froze the safe rewrites at -89% (0.37 -> 0.04). The gen4 checkpoint
reads 0.06, i.e. **-84%**. Barely moved. The semantic-distance gradient survives essentially intact,
so "graded, not confined" still stands as the honest preservation story.

### What this means
Two operating points, both reportable:

- **exp080 r2 s120** — the erasure row. Competitive with T2VUnlearning (-75.8% vs -73.4%) at severe
  motion cost.
- **exp110 s140** — the quality row. Much better video, erasure roughly at NegPrompt-plus levels on
  Ring-A-Bell and clearly better than NegPrompt on Gen/I2P/SafeSora.

A Pareto pair is a more honest contribution than picking one and hiding the other.

## Status
- [x] Submitted and complete (4 jobs).
- [x] Read against exp102/exp084's old-checkpoint numbers — **erasure worse, quality better**.
- [ ] DOVER scored locally on these outputs.
- [ ] Human review — still not done, and now more important: is 0.150 on Gen visibly different?
- [ ] `docs/comparability_t2vunlearning.md` §4 rewritten as a two-operating-point table.
