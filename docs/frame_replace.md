# Frame-Replace: Supervised V-Prediction Unlearning Toward Edited Latents

This document describes the **frame_replace** unlearning method for the "fire" concept in
CogVideoX-5b. It is the reference behind `zml/precompute/frame_replace_precompute.py`
(target construction) and `zml/unlearn/unlearn_frame_replace.py` (training).

In its base form, and unlike the ESD / UnHype family (see [`unhype.md`](unhype.md)), frame_replace
uses **no teacher, no classifier-free guidance, and no negative steering**. It is plain supervised
diffusion fine-tuning: take a clean latent, noise it, predict the velocity, regress against the true
velocity. The only twist is *what* the clean latent is — a fire-removed edit of the model's own
output. (An optional ESD-style variant, §4.4, adds a single frozen-teacher pass to soften that
target; it is off by default.)

---

## 1. Overview & motivation

For a fire prompt the model often produces a clip where fire appears only in **some** frames.
That observation is the whole basis of the method: if we can take such a clip and surgically
swap out just the fire-containing frames for fire-free ones from the *same* clip, we get a
target that is (a) almost identical to what the model already produces — so it stays on the
model's own distribution and minimizes collateral damage — and (b) fire-free. Fine-tuning the
model to map the fire prompt onto that edited target teaches it to stop generating fire while
disturbing as little else as possible.

The method has two stages:

1. **Precompute (offline).** Generate videos, detect fire per frame, build the edited
   fire-free latent `x0_edited`, and save it. Done once.
2. **Train (online).** Load the precomputed targets and run supervised v-prediction SFT on a
   PEFT LoRA.

Splitting it this way is a performance necessity: generating + decoding + running the fire
detector inside every training step would be far too expensive. The expensive, non-differentiable
work (sampling, VAE decode, detection) is paid once up front; the training loop only loads
tensors.

---

## 2. Latent geometry

Both stages share CogVideoX-5b's latent geometry at 49 pixel frames / 480×720:

```
latent shape (B, C, F, H, W) = (1, 16, 13, 60, 90)
```

The CogVideoX 3D causal VAE compresses time by a factor of 4, but with a causal anchor:
**latent frame 0 encodes exactly 1 pixel frame; every later latent frame encodes 4 pixel
frames**. Hence

```
num_pixel_frames = 1 + 4 · (num_latent_frames − 1) = 1 + 4·12 = 49
```

This 1+4k mapping is why the editing and fire-masking happen at *latent*-frame granularity but
the fire detector runs on *pixel* frames (see §3.2).

---

## 3. Stage 1 — building the edited target (precompute)

`zml/precompute/frame_replace_precompute.py`. For each `(prompt, seed)` in the prompt CSV:

### 3.1 Generate a clean latent

The pipeline runs a full sampling loop (`num_inference_steps ≥ 50`, `output_type="latent"`),
returning the **clean** scaled latent `x0` (i.e. `z_0`, the fully-denoised endpoint — not an
intermediate noisy state). The scheduler must be `v_prediction`, asserted up front so the target
matches what the trainer expects.

### 3.2 Detect fire, per frame → per latent frame

The latent is VAE-decoded to pixel frames and passed to `VideoFireDetector`, which returns a
per-pixel-frame fire confidence. A pixel frame counts as fire if `confidence ≥
frame_fire_threshold` (default 0.5). These pixel-frame flags are lifted to latent frames with
the 1+4k mapping:

> A latent frame is "fire" if **any** of the pixel frames it encodes contains fire.

This `any` is deliberately conservative — because a single latent frame bundles up to 4 pixel
frames, marking it fire-free requires *all* of its pixel frames to be fire-free, so no fire
leaks through the edit.

### 3.3 Replace fire frames by interpolating between bracketing donors

`edit_latent` replaces each fire latent frame along the `F` axis with a fire-free target that
**preserves motion across the fire block**. For a fire frame `i` bracketed by fire-free frames on
both sides, it linearly interpolates in latent space between the nearest fire-free frame before
(`lo`) and after (`hi`) it:

```python
w = (i - lo) / (hi - lo)
edited[:, :, i] = (1 - w) * latent[:, :, lo] + w * latent[:, :, hi]
```

Fire frames at the clip start/end have a fire-free neighbour on only one side and fall back to a
one-sided copy of it. The result is `x0_edited`: the model's own clip with its fire frames replaced
by a smooth fire-free ramp.

> **Why not just copy the nearest donor?** The original method hard-copied the single nearest
> fire-free frame into *every* fire frame, so a contiguous fire block became several *identical
> frozen* frames. SFT then learned "hold still," which suppressed motion globally — exp055 measured
> the eta=2 model losing 84% of concept motion and, tellingly, 29–43% on prompts with no fire to
> remove. Interpolating across the block keeps a plausible trajectory, so the target no longer
> teaches stillness. (Caveat: latent-space lerp is not pixel-linear, so a *long* fire block becomes
> a slow cross-fade rather than true motion — still far better than a freeze, and long all-fire
> spans are skipped by `min_nofire_frames`.)

### 3.4 Skipping & verification

A clip is **skipped** (and recorded in `skipped.json`) when:

- `no_fire` — no fire was detected, so there is nothing to unlearn from it; or
- `insufficient_donor_frames` — fewer than `min_nofire_frames` (default 2) fire-free latent
  frames exist, which would force the edit to copy one frame across most of the clip and yield a
  near-static, low-quality target.

In the same pass the script optionally decodes **both** the pre-edit and post-edit latents to
MP4 and re-runs the detector on the edited frames, so you can confirm the edit actually removed
fire — all from a single seeded generation, avoiding drift between two separate runs.

Outputs land in the run's `outputs_{timestamp}` directory: `latents/*.pt` (the `x0_edited`
tensors), `metadata.json` (one entry per kept target: `prompt`, `seed`, `latent_path`,
`scaling_factor`, donor map, …), `skipped.json`, and optionally `videos/`.

---

## 4. Stage 2 — supervised v-prediction training

`zml/unlearn/unlearn_frame_replace.py`. The base transformer is **frozen**
(`requires_grad_(False)`); a PEFT LoRA is attached to the attention projections
`["to_q", "to_k", "to_v", "to_out.0"]` and is the only thing trained. Gradient checkpointing is
enabled to fit the 5B model.

### 4.1 Setup done once

- **Prompt embeddings.** Each unique prompt's **T5** embedding is precomputed and cached
  (`do_classifier_free_guidance=False` — this method is CFG-free, so only the conditional
  embedding is needed).
- **Rotary embeddings (RoPE).** Built once from the fixed latent geometry, because the
  transformer does **not** compute them internally. Evaluation generates *with* RoPE, so training
  must supply the same positional regime — otherwise the LoRA would waste capacity correcting a
  train/eval positional mismatch.

### 4.2 The training step

Each step samples one target and performs a standard v-prediction update:

```python
entry = random.choice(metadata)            # one (prompt, edited-latent) target
x0    = load(entry.latent_path)            # x0_edited: the fire-free clean latent
emb   = prompt_emb_cache[entry.prompt]     # cached T5 embedding of the fire prompt

t        = randint(timestep_min, timestep_max)      # random diffusion timestep
noise    = randn_like(x0)
x_t      = scheduler.add_noise(x0, noise, t)        # forward diffusion to level t
v_target = scheduler.get_velocity(x0, noise, t)     # the regression target

v_pred   = transformer(x_t, emb, t, rope).sample    # LoRA model's prediction
loss     = mse(v_pred, v_target)
loss.backward(); optimizer.step()
```

**`add_noise`** evaluates the forward diffusion closed form
`x_t = √(ᾱ_t)·x0 + √(1−ᾱ_t)·noise`, jumping directly to noise level `t` (no chain simulation).
Sampling `t` uniformly over `[timestep_min, timestep_max)` trains the model to denoise at every
level it will see at inference.

**`get_velocity`** computes the v-prediction target `v = √(ᾱ_t)·noise − √(1−ᾱ_t)·x0`. The
**v**-objective (rather than ε- or x0-prediction) keeps the regression target well-scaled across
*all* timesteps, giving stable, uniformly-sized gradients — and it is the objective CogVideoX
was trained with, so the LoRA augments the base model instead of fighting it. Critically,
`add_noise` and `get_velocity` receive the *same* `(x0, noise, t)`, so `x_t` and `v_target` form
a consistent input/target pair.

**`transformer(...)`** is the LoRA-adapted CogVideoX DiT producing its velocity estimate
`v_pred`. Because only the LoRA is trainable, `loss.backward()` produces gradients only for the
adapter; the 5B base stays frozen.

> **Layout note.** The scheduler keeps latents channels-first `(B, C, F, H, W)`; the transformer
> wants frames-first `(B, F, C, H, W)`. So `x_t` is permuted on the way in and `v_target` is
> permuted to match before the MSE. The loss is computed in `float32` for numerical stability
> even though the model runs in `bfloat16`.

### 4.3 Checkpointing & live evaluation

Every `save_interval` steps the LoRA is saved and `zml/unlearn/eval.py::evaluate` runs over
three control prompt sets — **concept** (fire), **related**, and **unrelated** — reporting
`fire_detection_rate` (does it still produce fire?) plus quality/fidelity metrics
(`clip_score`, `colorfulness`, DOVER technical/aesthetic). The split lets us separate *successful
erasure* (concept fire rate drops) from *collateral damage* (related/unrelated quality should
stay flat).

Metrics are mirrored to wandb + mlflow and to the plain `metrics.jsonl` / `summary.json` files
via `MetricsRecorder` (see the metrics-logging note in the project `CLAUDE.md`).

### 4.4 Optional: ESD-style interpolated erase target (`erase_esd_eta`)

The plain erase branch regresses the fire prompt **all the way** to the donor (edited fireless)
target and drives the loss toward 0. But that specific donor latent is not our true goal — it is
just one fire-free clip, and pinning the loss to 0 means memorizing it. What we actually want is to
push the fire prompt *toward* fireless and stop at a sensible midpoint.

Setting `erase_esd_eta` (`η`) borrows the ESD trick of regressing toward a linear blend of the
model's own current prediction and the goal, rather than the goal alone. Per erase micro-step we do
one extra **frozen-teacher** forward — the same noised fire latent + fire prompt, but with the LoRA
adapter disabled and under `no_grad` — to get the base model's prediction `teacher`, then form

```
target = teacher − η · (teacher − donor) = (1 − η) · teacher + η · donor
```

and MSE the student against that (detached) target. The blend is computed in whichever space
`erase_loss_space` selects, so `teacher` is the base **velocity** (velocity space) or the base
**predicted-x0** (x0 space), matching `pred` and the donor term.

- `η = 1` → the plain donor target (identical to the base method; the teacher pass is redundant).
- `η = 0` → a no-op (target == base prediction; nothing is unlearned).
- `0 < η < 1` → partial redirection: the loss can settle at a genuine midpoint instead of
  overfitting the donor, trading erasure strength for lower collateral risk.

Only the **erase** branch uses `η`; the retention branch always regresses fully to its anchor
latent. `erase_esd_eta` is `None` by default, so existing configs are unaffected. Cost: one extra
no-grad base forward per erase micro-step (the teacher shares the LoRA weights with the adapter
disabled, so no second model is loaded). Swept in `exp053_frame_replace_esd_eta`, with
`exp051` serving as the `η = 1` reference.

### 4.x Choosing the retention set — it must be *disjoint* from the concept

The retention branch regresses fully to its anchor latents, so those anchors are a hard statement of
"keep producing this". That makes their **content** a design decision of the same weight as `η`, and
getting it wrong silently cancels the erase term.

**exp085 is the negative result that establishes this.** It ran the same eta grid as exp086 but on
nudity-specific anchors (exp079) instead of fire-era ones (exp041), and **every arm erased worse**.
The anchor set it actually trained on — exp079's `metadata_human_filtered.json`, 20 entries — is
**11/20 exposed-skin wardrobe**: swimwear ×4, leotard, sports bra, pyjamas, towels ×2, a
bare-shoulders close-up, a midriff close-up. The retention loss was pulling toward keeping exposed
torsos while the erase loss pushed away from the same features. exp041's fire anchors share nothing
with the concept region, so they never pull back — the "wrong" set won because it was the only one
not competing.

Two failure modes are worth naming because both are easy to repeat:

1. **Category labels are not a composition audit.** Three of those eleven sit in categories *named*
   `closeup_clothed` and `multiperson_clothed`. Scan the prompt text.
2. **Human filtering can skew composition.** Against exp079's source CSV the filter took medical
   4→1, parenting 2→1, bathing 3→1, while swimwear kept 4 of 5 — skin-heavy prompts render more
   reliably, so selecting on visual quality drifted the set skin-ward by accident. **Filter within
   category and preserve the balance**; regenerate a category rather than let it shrink.

The rule this produces:

> **A training retention set must be semantically disjoint from the concept. Concept-*adjacent*
> content is an evaluation instrument, not a training anchor.**

Swimwear, medical and clothed-intimacy prompts belong in a held-out `related` column, where
destroying them is collateral damage we measure and report — not in the retention set, where they
fight the objective. This generalizes: the same temptation exists on the face thread (protecting
"the other four celebrities" while erasing one) and the object thread (PSR's nine preserved classes
are scored, not trained toward).

Disjoint is necessary but not sufficient, or exp041 would already be the answer. Retention only
helps where the erase term does damage, so the anchors should keep the **shot grammar** of the
training targets — the same framings, settings, subject counts and motion — while differing in the
concept itself. That is what `prompts/cogvideox_nudity_retention_clothed.csv` (exp104) does, and
what exp105 tests against both exp041 and exp079.

---

## 5. How it compares to ESD / UnHype

| | ESD / UnHype | frame_replace |
|---|---|---|
| Teacher / CFG | yes (negative guidance toward a mapping concept) | none |
| Target | steered noise prediction `ε_target` | edited clean latent `x0_edited` |
| Supervision | guided regression / gradient matching | plain MSE on velocity |
| Targets built | on the fly each step | precomputed offline, once |
| Trainable params | LoRA (or hypernetwork) | LoRA on attention projections |

The trade-off: frame_replace is simpler and cheaper per step, and its target stays maximally
close to the model's own distribution (low collateral risk), but it can only unlearn from clips
where fire is *partial* — a fully-on-fire clip has no donor frame and is skipped. It is therefore
best seen as a targeted, distribution-preserving complement to the steering-based methods rather
than a drop-in replacement.

## Colorfulness is not a quality metric — a correction (2026-08-22)

Recorded because it steered several experiments wrongly.

Across exp123-exp136 the erasure/quality trade was tracked using **colorfulness recovery toward
base (36.3)** as the proxy for "the model got its appearance back". That proxy is invalid:

- Colorfulness is an unbounded saturation statistic. It has no upper penalty, so runs that
  **oversaturate** score as "recovered" or better (exp124 s200 reads 53.5, exp125 reads 49-75, both
  far above base) when they are in fact worse.
- It measures nothing about **sharpness** or naturalness, which is exactly where the high-eta arms
  fail. Human review of exp124/exp136 clips: *"oversaturated, weird, not-sharp"*.

DOVER-technical, which does measure technical quality, ranks the checkpoints the way human review
does and the way colorfulness did not. Low-rate checkpoints (rate <= 0.10), sorted by sharpness:

| checkpoint | eta | rank | rate | DOVER-t | DOVER-a | colour |
|---|---|---|---|---|---|---|
| base | — | — | 0.414 | 0.0700 | 0.8700 | 36.3 |
| **exp080 r2 s120** | 2 | 8 | 0.000* | **0.0643** | 0.8418 | 21.9 |
| **exp110 s140** | 2 | 8 | 0.000* | **0.0616** | **0.8871** | 35.4 |
| exp086 r? s80 | 1.0 | 8 | 0.010 | 0.0574 | 0.7435 | 22.7 |
| exp124 r1 s160 | 4 | 8 | 0.030 | 0.0443 | 0.7167 | 32.0 |
| exp124 r1 s140 | 4 | 8 | 0.000 | 0.0420 | 0.7036 | 25.6 |
| exp124 r1 s100 | 4 | 8 | 0.030 | 0.0337 | 0.2823 | 18.9 |
| exp124 r1 s60 | 4 | 8 | 0.000 | 0.0239 | 0.1572 | 14.3 |

*n=10 subset rates; full-set values are 0.100 (exp080) and 0.150 (exp110).

**The split is by eta, cleanly.** Every eta<=2 checkpoint sits at DOVER-t 0.057-0.064 (81-92% of
base); every eta>=4 checkpoint sits at 0.024-0.049 (34-70%). Raising eta buys erasure depth and
pays for it in sharpness, monotonically. That trade was invisible while colorfulness was the
quality axis, because colorfulness *rises* along the same direction.

**Rule going forward:** quality claims use DOVER-technical (sharpness) and DOVER-aesthetic
(naturalness). Colorfulness may only be reported as |colour - base|, and only as a saturation
diagnostic — never as evidence of recovery. Human review outranks all of them
([[feedback-detector-metrics-not-ground-truth]]); it caught this before the metrics did.

## Why the crude old dataset erases better than the realistic new one (2026-08-22)

exp080 (34 gen1-gen3 targets, baggy unrealistic wardrobe) still produces the best checkpoint, beating
exp110/exp123/exp124/exp136 trained on the deliberately realistic gen4 sets. Measured with
`tools/analyze_edit_directions.py`, which summarises each clip's edit scene-invariantly as its mean
LAB shift and asks how much survives averaging across the dataset — the component a low-rank adapter
can actually learn.

**Two hypotheses, both refuted by the measurement:**

| | edit magnitude | coherence |
|---|---|---|
| OLD-31 (exp080) | 6.2 | 0.538 |
| GEN4-100 (exp110) | **13.2** | **0.605** |
| CLEAN-75 (exp123/124/136) | **14.7** | **0.615** |

gen4's edits are more than **twice as large** and slightly **more coherent**. So "fitted donors give a
small push" (the premise behind raising eta) and "wardrobe diversity cancels the shared direction"
are both wrong as stated — gen4 has more shared signal, not less.

**What actually differs is the *content* of that shared direction.** Decomposing the mean LAB shift
into luminance and chroma:

| dataset | luma dL | chroma | **chroma/luma** |
|---|---|---|---|
| OLD-31 | 0.63 | 0.93 | **1.47** |
| GEN4-100 | 3.11 | 1.49 | 0.48 |
| CLEAN-75 | 3.70 | 1.62 | 0.44 |

The old dataset's learnable direction is **chroma-dominated**: it removes warm skin tones roughly 1.5x
as much as it darkens. That is semantically the concept — skin is a chroma signature. gen4's is
**luminance-dominated** by 2:1 the other way: mostly "make it darker", because fitted garments in
deliberately varied colours (mustard, emerald, rust, turquoise) push chroma in different directions
that partially cancel, while *every* garment darkens relative to bare skin, so darkening survives
averaging.

A rank-r adapter learns the surviving direction. Train on gen4 and it learns **darken**, which is a
global style edit, not an erasure — and that is exactly what independent evidence already showed:
exp113 measured the gen4 checkpoint shifting colour on prompts with no nudity in them (53.1 vs base
45.8), i.e. the model applying the learned style everywhere. It also explains why raising eta made
things worse rather than better: eta amplifies the shared direction, so a luminance-dominated one
amplified into the oversaturated, unsharp clips human review rejected in exp124/exp136.

**Design rule this yields — it inverts gen4's premise.** Build the dataset so the *shared* edit is
the concept-relevant one, which means holding the donor wardrobe's colour family roughly constant so
chroma shifts reinforce instead of cancelling. Realism per garment is fine; deliberate colour variety
across the set is what destroyed the signal. Note this also predicts gen4's higher yield (50% vs 26%)
was bought at the cost of the property that matters.

**Untested prediction, cheap to check:** a colour-homogeneous subset of gen4 should have a higher
chroma/luma ratio than the full set, and should erase better at matched size. `analyze_edit_directions.py`
scores any candidate subset in about a minute, so a gen5 prompt set can be validated *before* any GPU
time is spent on it.
