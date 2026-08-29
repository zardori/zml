---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  Tests whether `erase_esd_eta` is a reparameterization of `retention_weight`. AdamW is invariant to
  a global gradient rescale, and at LoRA init (B zero-init, so pred == teacher) the erase gradient is
  exactly eta times the eta=1 gradient -- so eta cannot act as a step size. Its only remaining channel
  is the ratio r = eta/w against the retention branch, which does not scale with eta. That predicts
  (eta=2,w=1) == (eta=1,w=0.5) and (eta=1,w=1) == (eta=2,w=2). 2x2 grid, 60 steps, read in WEIGHT
  SPACE against exp080 run_002 and exp086 run_002 (identical seed/data/RNG stream). Not yet submitted.
---

# exp178 — is eta redundant with retention_weight?

## Where this came from

While checking the algebra of the ESD-style erase target, a sharper question surfaced. The target is

    target = (1 - eta) * teacher + eta * donor

and `eta = 1` makes the teacher cancel exactly — `(1-1)*T + 1*D = D` — i.e. **plain masked SFT toward
the donor**, at the cost of a wasted teacher forward pass. `_sft_velocity_loss`'s own docstring says
as much. So the "SFT baseline" everyone assumes is missing has in fact been run three times:

| run | eta | w | dataset | retention |
|---|---|---|---|---|
| exp085 run_002 | 1.0 | 1.0 | exp080 | exp079 nudity (skin-heavy — confounded) |
| **exp086 run_002** | **1.0** | **1.0** | **exp080** | **fire (exp041)** |
| exp146/exp148 run_001 | 1.0 | 1.0 | gen5 | fire |

exp086 run_002 is exp080 run_002 verbatim except eta, and it lost.

## Why that loss proves nothing

Write `u = pred - teacher` and `d = donor - teacher`. The erase residual is `u - eta*d`, so the erase
gradient is `∝ (u - eta*d)`. LoRA B is zero-initialised, so at step 0 `u = 0` **exactly** and the
gradient is `-eta*d` — precisely `eta` times the `eta=1` gradient, same direction, longer.

AdamW normalises by `m / sqrt(v)`, both of which scale with the gradient. **A global rescale of the
gradient produces an identical AdamW update.** So eta cannot be acting as a learning rate.

What it *can* act on is the ratio against retention. The two branches are backpropped separately into
the same `.grad` buffer (`unlearn_frame_replace.py`), and the retention branch carries `w` and does
not scale with eta:

    total = eta * g_erase(1) + w * g_retain   ∝   g_erase(1) + (w/eta) * g_retain

Only `r = eta / w` survives. So exp086 run_002 did not lose because SFT is a weaker objective — it
lost because its erase term was weighted **half** as heavily as exp080 run_002's. Untested.

The equivalence is exact only at `u = 0` and degrades as the student moves (once `u != 0`,
`u - 2d` and `u - d` stop being parallel). **That decay rate is the empirical content of this run.**

## Design

2x2 over `erase_esd_eta: [1.0, 2.0]` x `retention_weight: [0.5, 2.0]` — ratios 2, 0.5, 4, 1.
Everything else is exp086 verbatim. Two cells complete matched-ratio pairs with runs already on disk:

| r | this grid | existing reference |
|---|---|---|
| 2 | eta=1, w=0.5 | exp080 run_002 (eta=2, w=1) — checkpoints already local, steps 20–200 |
| 1 | eta=2, w=2.0 | exp086 run_002 (eta=1, w=1) — needs `pull_results.sh --include-weights` |

### Grid ordering (verified against `submit_job.expand_grid`)

| run | eta | w | ratio | pairs with |
|---|---|---|---|---|
| **run_001** | 1.0 | 0.5 | **2** | **exp080 run_002** (eta=2, w=1) — checkpoints already local |
| run_002 | 1.0 | 2.0 | 0.5 | — (extends the ratio axis) |
| run_003 | 2.0 | 0.5 | 4 | — (extends the ratio axis) |
| **run_004** | 2.0 | 2.0 | **1** | **exp086 run_002** (eta=1, w=1) — weights need pulling |

run_001 is the primary test and needs nothing downloaded. Read it first:

    uv run python tools/compare_lora_weights.py \
      experiments/nudity/exp178_eta_retention_equivalence/grid_*/run_001/outputs \
      experiments/nudity/exp080_frame_replace_nudity_gen2/grid_20260806_211043/run_002/outputs \
      --steps 20,40,60

## Readout is weight-space, not metrics

`lora_dropout: 0.0`, `global_seed: 42`, `gradient_accumulation_steps: 4` are shared by all six runs,
and evaluation uses an isolated `torch.Generator` per prompt (`zml/unlearn/eval.py`), so **the
training RNG stream is bit-identical across runs** — same data order, same timesteps, same noise.
Weight differences are therefore attributable to (eta, w) alone.

`tools/compare_lora_weights.py` reports relative L2 and cosine similarity between checkpoints.
Two calibration scales, both from data rather than a guessed threshold:

- **different-ratio distance** — exp080 run_002 vs exp086 run_002 at matched steps (what a real eta
  difference looks like);
- **trajectory distance** — exp080 run_002 step20 vs step40 (how far a run moves on its own).

A matched-ratio pair counts as equivalent only if it sits far below both.

`eval_num_prompts: 1` is a smoke test for a silently diverged run. **Do not rank on it.**

## Decision rule

- **Pairs agree through step 60** → eta is redundant. Drop `erase_esd_eta` from the method and the
  paper's hyperparameters, delete the teacher forward pass (1 of every 3 forwards in the erase
  branch), and re-read every eta sweep to date as a retention sweep. Then extend the r=2 cell to 200
  steps with exp086's eval settings to confirm at the checkpoint we actually select.
- **They diverge** → the `2D - T` overshoot has content beyond reweighting, and the step at which
  they separate dates when the extrapolation begins to matter. That is the justification for keeping
  the ESD term, which we currently do not have.

Either way the eta column in our tables stops being unfalsified.

**Cross-thread, flag but do not act on:** imagenet and face_identity hardcode eta=2.0 with w=1.0 in
every frame_replace config. If eta is redundant here it is redundant there.

## Calibration (measured, before submitting anything)

`tools/compare_lora_weights.py` compares **`lora_B` only** by default. That is not a detail: PEFT
initialises A randomly and B at exactly zero, so B is the learned update and A is a shared constant.
Measured on exp080's lr grid at step 20 — `||A||` = 21.19 / 21.27 / 21.50 / 22.27 across a **10x**
learning-rate range, while `||B||` = 0.73 / 1.51 / 2.99 / 6.14. Comparing full adapters buries the
signal under ~93% shared init and compresses every distance toward zero. (A first pass did exactly
that and made a 2x lr change look like a 1.8% effect.)

Two scales, both measured from runs already on disk, not guessed:

| comparison | rel L2 | cosine | \|\|B\|\| ratio |
|---|---|---|---|
| a checkpoint against itself | 0 | 1.00000000 | 1.00 |
| **trajectory**: run_002 step20 vs step40 | 1.206 | 0.918 | 2.06 |
| **trajectory**: run_002 step40 vs step60 | 0.499 | 0.933 | 1.28 |
| **2x lr** (1e-4 vs 2e-4), step 20 | 1.023 | 0.977 | 1.98 |
| **2x lr** (1e-4 vs 2e-4), step 40 | 0.818 | 0.873 | 1.53 |
| **2x lr** (1e-4 vs 2e-4), step 60 | 0.917 | 0.803 | 1.50 |

**Cosine is the discriminating statistic.** A genuine 2x step-size change walks it 1.000 -> 0.977 ->
0.803 over 60 steps while keeping the direction broadly similar; matched-ratio pairs should instead
sit at ~0.999+ with a `||B||` ratio of ~1.0.

The lr grid also corroborates the premise this experiment rests on: `||B||` is **near-linear in the
learning rate** (0.73 : 1.51 : 2.99 : 6.14 against lr 0.5 : 1 : 2 : 5). That is the AdamW signature —
displacement is set by `lr`, not by gradient magnitude — which is precisely why eta, a pure gradient
rescale, cannot be acting as a step size.

Predicted signatures at step 60, so the result cannot be read after the fact:

- **eta redundant** (ratio hypothesis): cosine ~0.999+, `||B||` ratio ~1.0, rel L2 ~0.05 or below.
- **eta acts as a step size**: the lr signature — `||B||` ratio ~1.5-2.0, cosine ~0.80.
- **eta does something of its own**: neither; cosine drops without the norm ratio moving.

## Status

Configured, **not submitted** — project owners submit. 4 jobs, `slurm_time: "0-8:00:00"` each:

    ./submit_job.py helios experiments/nudity/exp178_eta_retention_equivalence/config.yaml

Optional, for the r=1 pair only (the r=2 pair is fully served by local files):

    ./pull_results.sh --experiment experiments/nudity/exp086_eta_ablation_fire_retention --include-weights

That pulls ~4 GB (6 runs x 20 checkpoints x 33 MB) for the ~100 MB actually needed, so it is worth
skipping unless the r=2 pair comes back ambiguous.
