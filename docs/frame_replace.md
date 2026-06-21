# Frame-Replace: Supervised V-Prediction Unlearning Toward Edited Latents

This document describes the **frame_replace** unlearning method for the "fire" concept in
CogVideoX-5b. It is the reference behind `zml/precompute/frame_replace_precompute.py`
(target construction) and `zml/unlearn/unlearn_frame_replace.py` (training).

Unlike the ESD / UnHype family (see [`unhype.md`](unhype.md)), frame_replace uses **no teacher,
no classifier-free guidance, and no negative steering**. It is plain supervised diffusion
fine-tuning: take a clean latent, noise it, predict the velocity, regress against the true
velocity. The only twist is *what* the clean latent is — a fire-removed edit of the model's own
output.

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

### 3.3 Replace fire frames with the nearest donor

`edit_latent` replaces each fire latent frame along the `F` axis with the **nearest** fire-free
("donor") latent frame from the same clip:

```python
donor = min(nofire_frames, key=lambda j: abs(j - i))   # nearest fire-free frame
edited[:, :, i] = latent[:, :, donor]
```

Choosing the *nearest* fire-free frame keeps the edit temporally local, so the patched clip
stays as close as possible to the original motion/content. The result is `x0_edited`: the
model's own clip with its fire frames overwritten by neighboring fire-free ones.

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
