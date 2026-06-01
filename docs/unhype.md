# UnHype: CLIP-Guided Hypernetworks for Dynamic LoRA Unlearning

> Wójcik, Petrenko, Gromski, Spurek, Zięba.
> *UnHype: CLIP-Guided Hypernetworks for Dynamic LoRA Unlearning.*
> arXiv:2602.03410 — https://arxiv.org/abs/2602.03410

This document summarizes the UnHype method and its training objective as presented in the
paper (which targets **text-to-image** diffusion), and closes with a short note on adapting it
to our **text-to-video** setting (CogVideoX-5b). The conceptual content here is the reference
behind the implementation in `zml/unlearn/unhype.py` and `zml/unlearn/unhype_modules.py`.

---

## 1. Overview & motivation

Classic LoRA-based unlearning (e.g. ESD) trains **one fixed LoRA adapter per erased concept**.
This is a static per-concept bottleneck: each new concept needs its own training run and its own
stored weights, and a single adapter does not generalize to synonyms or related concepts.

UnHype replaces the fixed adapter with a **hypernetwork** that *generates* LoRA weights on the
fly, conditioned on the prompt. At inference the model receives a prompt, embeds it with CLIP, and
the hypernetwork produces the LoRA weights to apply for that prompt. This yields:

- **Context-aware unlearning** — weights depend on the actual prompt embedding, so the method
  generalizes to unseen synonyms and semantically related concepts.
- **Scalable / multi-concept unlearning** — a single hypernetwork covers many concepts instead of
  one adapter each.
- **A built-in "semantic switch"** — for *safe* (non-target) prompts the hypernetwork outputs
  near-zero weights, leaving the base model untouched.

The framework plugs into both Stable Diffusion and modern flow-based T2I models (Flux), and is
demonstrated on object erasure, celebrity erasure, and explicit-content removal.

---

## 2. Architecture

The hypernetwork is an **MLP** `H_φ` that maps a (concept embedding, step) pair to a full set of
LoRA weights:

```
θ_s = H_φ(c, s)
```

- **`c`** — a 768-d **CLIP text embedding** of the concept/prompt.
- **`s ∈ [0, S]`** — a *continuous optimization-step* variable; the hypernetwork is conditioned on
  where along an (implicit) unlearning trajectory we are.
- **Output `θ_s`** — the flattened LoRA weight vector for all target modules at step `s`
  (cross-attention projections for SD; value/output projections for Flux).

The design is inspired by **Hypernet Fields**: rather than predicting only the *final* converged
LoRA, `H_φ` models the **entire optimization trajectory** `s ↦ θ_s`. This is what makes the
gradient-matching training objective below possible.

---

## 3. Training objective

UnHype is trained with two supervised losses (task loss enters only through its gradient).

### 3.1 Task loss — ESD-style steering

The task loss recasts unlearning as guided regression: with the predicted weights applied, the
model's noise prediction for the target concept `c` should match a **steered target** that pushes
away from `c` and toward a benign *mapping* concept `c_m` (e.g. "forest" instead of "fire"):

```
ℒ_task = E ‖ ε_{θ* + θ_s}(z_t, t, c) − ε_target ‖²

ε_target = ε_{θ*}(z_t, t, c_m) − γ · ( ε_{θ*}(z_t, t, c) − ε_{θ*}(z_t, t, c_m) )
```

- `θ*` — frozen base-model weights; `θ_s` — hypernetwork-predicted LoRA at step `s`.
- `c_m` — mapping concept the target should be redirected to.
- `γ` — repulsion strength (negative-guidance scale): larger `γ` pushes harder away from `c`.

### 3.2 Removal loss — gradient matching (Hypernet Fields)

Instead of explicitly running and storing an SGD trajectory, UnHype supervises the hypernetwork so
that its **predicted step** matches a single **simulated SGD step** on the task loss:

```
ℒ_remove = ‖ ( H_φ(c, s+1) − H_φ(c, s) ) + η ∇_{θ_s} ℒ_task ‖²
```

- `Δθ_pred = H_φ(c, s+1) − H_φ(c, s)` — the step the hypernetwork predicts along the trajectory.
- `Δθ_task = −η ∇_{θ_s} ℒ_task` — the target step: one gradient-descent update of the task loss.
- `η` — **simulated learning rate** (a hyperparameter, ~`1e-3`…`1e-4`).

This aligns the hypernetwork's trajectory with the unlearning task's gradient field without ever
precomputing or storing the final per-concept weights.

### 3.3 Retain loss — the semantic switch

To avoid catastrophic forgetting of non-target concepts, the hypernetwork is pinned to its
zero-weight initialization for *retain* concepts:

```
ℒ_retain = E ‖ H_φ(c_retain, s) − H_φ(c_retain, 0) ‖²
```

Since `H_φ(·, 0) ≈ 0`, this keeps `θ_s ≈ 0` for safe prompts — i.e. the base model is preserved
unchanged whenever the prompt is not a target concept.

### 3.4 Total objective

```
ℒ = λ_remove · ℒ_remove + λ_retain · ℒ_retain
```

`λ_remove`, `λ_retain` weight the two terms. The task loss contributes only via its gradient inside
`ℒ_remove`.

---

## 4. Simulated unlearning trajectory

Each training step:

1. Sample an unlearning step `s ∼ U(0, S)` (the paper uses **S = 300** across experiments).
2. Predict `θ_s = H_φ(c, s)`.
3. Compute `ℒ_task` with `θ_s` applied and backprop to get `∇_{θ_s} ℒ_task`.
4. Form the target SGD step `−η ∇_{θ_s} ℒ_task` and enforce
   `H_φ(c, s+1) − H_φ(c, s) ≈ −η ∇_{θ_s} ℒ_task` via `ℒ_remove`.
5. Add `ℒ_retain` for sampled retain concepts and update `φ`.

There is **no explicit meta-gradient** and no precomputed per-concept LoRA modules: the trajectory
is supervised by ordinary task-loss gradients evaluated at the sampled step (forward-mode gradient
matching). This is what removes the static per-concept bottleneck.

---

## 5. Inference

The final weights for a prompt are produced in a **single forward pass** at the trajectory
endpoint:

```
θ_S = H_φ(c, S)
```

Generation then proceeds with a modified classifier-free guidance that applies the generated
weights **only to the conditional branch**, keeping the unconditional/base path frozen (Stable
Diffusion):

```
ε_CFG = (1 + w) · ε_{θ* + θ_S}(z_t, t, c) − w · ε_{θ*}(z_t, t, c_0)
```

For Flux the generated weights are applied directly to the model for the whole sampling process.
For safe prompts `θ_S ≈ 0`, so the base model's behavior is recovered automatically.

---

## 6. Adapting to text-to-video (CogVideoX-5b) — project note

The paper targets text-to-image diffusion. Bringing UnHype to our CogVideoX-5b setting requires a
few adjustments (already reflected in `zml/unlearn/unhype.py`):

- **Video latents.** The denoising loss keeps the same form, but `z_t` is a CogVideoX video latent
  (channels × frames × H × W) rather than an image latent. The ESD-style steered target `ε_target`
  is computed identically with the frozen base transformer.
- **Two text encoders.** CogVideoX conditions generation on a **T5** text embedding, whereas the
  hypernetwork is conditioned on a separate **CLIP** embedding of the prompt (configured via
  `clip_model_id`). T5 drives the diffusion model; CLIP only feeds `H_φ`.
- **LoRA target modules.** The generated weights attach to the transformer attention projections
  `["to_q", "to_k", "to_v", "to_out.0"]`, consistent with the existing ESD/LoRA setup.
- **Live evaluation.** Training reuses `zml/unlearn/eval.py`; the hypernetwork integrates through a
  `prepare_for_prompt` callback that predicts and injects `θ_S` for each evaluation prompt before
  generation, so eval measures the dynamically generated adapter.

These are project-specific choices, not part of the original paper; treat this section as a bridge
between the published method and our implementation.
