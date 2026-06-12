# Applying UnHype (CLIP-guided hypernetwork) to CogVideoX — progress notes

**Scope:** experiments **exp016–exp031**, our attempts to port the UnHype hypernetwork
unlearning method (see [`unhype.md`](unhype.md)) from text-to-image diffusion to our
text-to-video setting (CogVideoX-5b), targeting the **"fire"** concept.

**Headline:** none of the online-hypernetwork runs achieved clean unlearning. But the
sequence is *not* a flat list of failures — each run localized a specific obstacle, and the
later runs (exp030/exp031) pin down *why* a method that works for image erasure is hard to
adopt for video. This document groups the ~16 runs into the **distinct approaches** we tried,
with aggregated metrics for each.

---

## How to read the metrics

Each run periodically generates videos for held-out **control** prompt sets and scores them:

- **Fire-detection rate (FDR)** — fraction of *concept* (fire) videos in which our fire
  detector still fires. **Lower = more erasure.** This is the primary unlearning signal.
- **CLIP (concept)** — mean CLIP text–video alignment on the *fire* prompts. A drop here can
  mean erasure *or* general quality collapse, so it must be read together with FDR/colorfulness.
- **CLIP (unrelated)** — mean CLIP alignment on *unrelated* prompts. This is the **collateral
  damage / "semantic switch"** guard: it must stay near baseline, otherwise we are degrading the
  base model rather than removing one concept.
- **Colorfulness** (later runs) — guards against the desaturation-collapse failure mode (FDR
  drops only because the video turned grey, not because fire was replaced). Wildly high values
  (~100) mean noise/garbage output (divergence).

**Reference baseline** (base CogVideoX, same n=5 eval prompts the hypernet runs use):
`concept FDR ≈ 1.0`, `concept CLIP ≈ 0.326`, `unrelated CLIP ≈ 0.33`, `concept colorfulness ≈ 22.6`.
(A direct ESD LoRA — exp006 — *does* erase fire on this setup; see exp030 below. So the target is
known to be achievable; the question is whether the hypernet can learn it.)

---

## All runs at a glance (final-step eval)

| exp | approach | step | **FDR (fire)** | **CLIP fire** | **CLIP unrelated** | verdict |
|-----|----------|------|:----:|:----:|:----:|---------|
| 016 | A. direct port | 300 | 1.0 | 0.32 | 0.33 | no change |
| 019 | A. diagnostics | 100 | 1.0 | 0.33 | 0.33 | no change (confirms dead-init) |
| 020 | A. gentle | 2000 | 1.0 | 0.33 | 0.33 | no change |
| 021 | A. moderate | 2000 | 1.0 | 0.33 | 0.33 | no change |
| 022 | A. aggressive | 2000 | 1.0 | 0.33 | 0.33 | no change |
| 023 | B. calibrated (MSE) | 2000 | 1.0 | 0.325 | 0.330 | no-op adapter |
| 024 | B. bigger sim-lr (MSE) | 500 | 1.0 | 0.328 | 0.330 | no-op adapter |
| 025 | C. cosine loss | 500 | 1.0 | 0.330 | 0.330 | moves, wrong direction |
| 026 | C. stronger guidance | 1000 | 1.0 | 0.324 | 0.329 | direction stuck |
| 027 | C. timestep var-reduction | 400 | 1.0 | 0.323 | 0.329 | direction stuck |
| 028 | D. distill control | 2000 | 1.0 | 0.327 | 0.331 | inconclusive (optimizer) |
| 030 | D. static-apply control | 300 | **0.6** | 0.280 | 0.322 | **path proven sound** |
| 029 | E. prompt var + step-fix | 150 | 0.0\* | 0.144\* | 0.162\* | diverged (garbage) |
| 031 | F. stabilized cosine | 150 | **0.4** | 0.220 | 0.231 | **best — then leaks** |

`*` exp029 "FDR 0.0" is meaningless: the run diverged (θ-norm → 1853), so *all* output is noise
(CLIP collapsed on **both** concept and unrelated). It did not erase fire; it destroyed the model.

(exp017 = helios smoke test, exp018 = a 9-run scale grid that was **deferred**, never run — both
omitted from the metrics table.)

---

## The approaches, in order

### A. Direct port of the paper's online hypernet — *exp016, exp019, exp020–022*

**Idea.** Implement UnHype as published: an MLP hypernet `H_φ(c, s)` trained by Hypernet-Fields
**gradient matching** (`loss_remove = ‖(H(c,s+1)−H(c,s)) + η∇_θ L_task‖²`) plus a retain term,
with an ESD-style steered task loss. exp016 used the paper-like small setup
(`lora_rank=1`, `S=50`, `simulated_lr=1e-3`, short prompts); exp020–022 scaled capacity/length
and swept an "aggressiveness" axis (`simulated_lr` × `retain_weight`).

| run | FDR | CLIP fire | CLIP unrelated |
|-----|:---:|:---:|:---:|
| exp016 / 019 / 020 / 021 / 022 | 1.0 | 0.32–0.33 | 0.33 |

**Result.** Zero measurable change vs base, in every run.

**Insight (the dead-LoRA-init trap).** Diagnostics (exp019) showed the trajectory never left the
origin: `target_step = predicted_step = θ_S ≈ 0`. Root cause is a clash between two standard
choices: LoRA initializes `B = 0` (so `A·B = 0`, base preserved), and the hypernet zero-inits its
output layer (so it emits a *constant* adapter). Together `A = B = 0`, and because
`∂L/∂B ∝ A` and `∂L/∂A ∝ B`, the removal gradient is **exactly zero** — nothing can ever train,
independent of any hyperparameter. Fixed in `unhype_modules.py` (small nonzero output weight,
`B`-rows zeroed so `B = 0` exactly but `∂L/∂B ∝ A ≠ 0`). A second, milder issue: `simulated_lr`
was ~40–200× too small, so even with a live gradient the endpoint displacement (~0.03) was inert.

---

### B. Calibrating the gradient-matching (MSE) loss — *exp023, exp024*

**Idea.** With init fixed, make the MSE removal target actually move `θ`. exp023 raised
`simulated_lr` 0.005→0.3 (endpoint target `O(1)`); exp024 raised it again 0.3→30 to close a
measured **120× scale gap** between `predicted_step` (~0.13–0.36) and `target_step` (~0.002).

| run | FDR | CLIP fire | CLIP unrelated | θ_S norm |
|-----|:---:|:---:|:---:|:---:|
| exp023 | 1.0 | 0.325 | 0.330 | 15.16 (frozen at init) |
| exp024 | 1.0 | 0.328 | 0.330 | 15.16 (frozen at init) |

**Result.** No-op adapter; `θ_S` frozen at its init value, CLIP scores byte-identical across
checkpoints.

**Insight (MSE is trivially satisfiable either way).** An MSE between the predicted step and a
*tiny* online target has two cheap minima and neither erases: if `‖pred‖ ≫ ‖target‖`, the loss is
dominated by `‖pred‖²`, so its gradient just **shrinks the step to zero** → constant-in-`s`
trajectory → endpoint stuck at `θ(0)`. If you instead match the norms (exp024), the MSE is
satisfiable at `~1e-7` with *any* aligned-enough vector, so there is **no pressure to move off
init**. MSE conflates "right direction" with "right magnitude," and in both regimes the lazy
solution is to not erase.

---

### C. Cosine removal loss + stronger / less-noisy steering — *exp025, exp026, exp027*

**Idea.** Decouple *direction* from *magnitude*. New removal loss
`(1 − cos(pred, target)) + w·(‖pred‖−‖target‖)²` (exp025). Then attack the *target* itself:
exp026 raised `negative_guidance_scale` 1→3 (give the ESD steered target a larger, more consistent
erasure direction); exp027 averaged the target over **K=8 timestep snapshots** to cut variance.

| run | change | FDR | CLIP fire | CLIP unrelated | `loss_remove_direction` |
|-----|--------|:---:|:---:|:---:|:---:|
| exp025 | cosine loss | 1.0 | 0.330 | 0.330 | ~0.94–1.0 (no trend) |
| exp026 | guidance ×3 | 1.0 | 0.324 | 0.329 | ~1 (pinned) |
| exp027 | K=8 timestep avg | 1.0 | 0.323 | 0.329 | ~0.99 → 0.97 |

**Result.** The adapter sometimes *moved* (exp025: θ 15.16→17.06, videos visibly changed) — but in
a direction **unrelated to erasure** (fire restructured, not reduced). The alignment metric
`loss_remove_direction = 1 − cos(predicted_step, ESD target)` stayed **pinned near 1
(orthogonal)** in all three.

**Insight (the online ESD target direction is too high-variance to align to).** The hypernet's
step `θ_{s+1}−θ_s` is a *smooth, deterministic* function of `(CLIP, s)`. The removal target
`−η∇_θ L_task` is built from a **single** noisy sample — one (target, mapping) prompt pair, one
timestep `t`, one stochastic rollout — and its direction reshuffles every step. A deterministic
function cannot align to a target that has no stable direction. Reducing **timestep** variance
(exp027) barely helped (0.99→0.97), which pointed at the *other* dominant variance source — the
prompt (see E). This is the first real video-specific obstacle: image-erasure papers use short,
low-diversity prompts where the ESD target is nearly stationary; CogVideoX is conditioned on long,
detailed captions, so each pair yields a very different gradient direction.

---

### D. Controls — is the failure in *learning* or in *applying*? — *exp028, exp030*

Before burning more GPU-hours on the online signal we isolated the apply/eval path from the
learning signal, by feeding a **known-good** target: `θ*`, the flat LoRA vector of exp006's direct
ESD adapter, which *does* erase fire.

- **exp028 (offline distill):** train `MSE(H(c,S), θ*)`, no diffusion in the loop.
  **Inconclusive** — the optimizer never reached `θ*` (`distill_cosine` 0.0005→0.057). Mean-reduced
  MSE over 8.26M output elements gives ~4e-9 per-element gradients; Adafactor barely moves the
  4.2B-param output layer. An optimization-speed artifact of the control, not a finding about the
  method. FDR stayed 1.0 only because `θ*` was never emitted.

- **exp030 (static-apply, zero training):** inject `θ*` *directly* through the **same**
  `decode → apply_flat` path the hypernet uses, run one eval, stop.

  | set | FDR | colorfulness | CLIP |
  |-----|:---:|:---:|:---:|
  | concept (fire) | **0.6** (vs 1.0 base) | **9.0** (vs 22.6) | 0.28 (vs 0.326) |
  | unrelated | 0.0 | 24.95 | 0.322 (held) |

**Insight (the wiring is correct; the bottleneck is 100% the online signal).** Injected directly,
`θ*` erases fire (FDR and colorfulness drop, orange/red removed) while unrelated is cleanly
preserved. So `decode` layout, `alpha/rank` scaling, and eval conditioning are all sound. This
*decisively* removes the apply path as a suspect: every failure in approaches A–C is upstream, in
the **learning signal** that produces `θ`.

---

### E. Attacking prompt variance + the step embedding — *exp029*

**Idea.** Given C and D, target the *remaining* variance source (prompt) and a latent bug.
(1) `target_prompt_batch_size=4`: average the ESD target over a batch of prompt pairs per step.
(2) Step-embedding fix: `s` was feeding the sinusoid un-normalized, so high frequencies wrapped
~`S/2π` times and `θ_{s+1}−θ_s` spun with `s`; normalize `s` into a monotonic `[0, π]` phase.

**Result.** For the **first time the alignment gate moved**: `loss_remove_direction` fell
**1.13 → 0.55** (cos ≈ 0.45) over steps 0–49 — the success criterion that never budged in A–C.
Then it **diverged**: a magnitude feedback loop (`θ ↑ → loss_task ↑ → grad ↑ → target ↑ → θ ↑`)
blew `θ`-norm to **1853**, and the run produced pure noise.

| run | FDR | CLIP fire | CLIP unrelated | colorfulness | θ_S norm |
|-----|:---:|:---:|:---:|:---:|:---:|
| exp029 (step 150) | 0.0\* | 0.144\* | 0.162\* | ~97–100 (noise) | 1853 |

`*` divergence, not erasure — CLIP collapsed on **both** sets; the model was destroyed.

**Insight (the signal is alignable, but the coupled dynamics are unstable).** Reducing **prompt**
variance was the missing piece that exposed an alignable ESD direction — confirming the C
diagnosis. But the online formulation couples adapter magnitude to the target it chases, which is
an unstable positive feedback loop with no native damping.

---

### F. Stabilizing the converging signal — *exp031*

**Idea.** Keep what made exp029's gate drop, kill the divergence, and fix one more bug.
(1) **Per-prompt conditioning fix:** exp029 conditioned the hypernet on only the *first* prompt
while averaging the target over *all* sampled prompts — a mismatch that caps the achievable
cosine. Now each prompt is conditioned and the per-prompt removal terms are averaged.
(2) Divergence guards: `target_step` norm cap, hypernet grad-norm clip, and an abort threshold.
(3) Prompt set extended 16 → 40 pairs.

**Result — our best run, and the clearest statement of the remaining obstacle:**

| step | FDR (fire) | CLIP fire | CLIP unrelated | θ_S norm |
|:----:|:----:|:----:|:----:|:----:|
| 50  | 0.8 | 0.305 | **0.333** (clean) | 16.7 |
| 100 | 0.6 | 0.295 | **0.330** (clean) | 19.2 |
| 150 | 0.4 | 0.220 | **0.231** (leaking) | 30.5 |

For the first ~100 steps this is **the behaviour we want**: fire FDR falls 0.8 → 0.6 → 0.4 while
unrelated CLIP stays at baseline (0.33) — genuine, prompt-selective erasure, stable (no
divergence; θ grew smoothly 16→30 instead of exploding). Then by step 150 the adapter starts
**damaging unrelated** generations too (unrelated CLIP 0.33 → 0.23): the prompt-conditioned
"semantic switch" does not hold as `θ` grows — the adapter drifts toward a *global* perturbation
rather than a fire-specific one.

**Insight (the core difficulty for video).** The hard part is not erasing fire (exp030 shows the
adapter capacity exists) and no longer the learning signal being un-alignable (exp031 shows it
converges). It is the **semantic switch**: keeping the generated adapter near-zero for non-target
prompts *while* it grows large enough to erase the target. In the image setting, low prompt
diversity makes target and non-target embeddings cleanly separable and the switch comes nearly for
free; with CogVideoX's long, overlapping captions the hypernet cannot keep the two regimes
separated as the erasure magnitude grows.

---

## Why the hypernet is hard to adopt for video — summary of insights

1. **Init clash gives an exactly-zero training signal** (A). Standard LoRA `B=0` + zero-init
   hypernet output → `A=B=0` → removal gradient is exactly 0. Easy to miss because nothing errors;
   it just never trains. (Fixed, but it masked everything for exp016–022.)

2. **MSE gradient matching has no useful minimum here** (B). Against a tiny, noisy online target,
   MSE is minimized either by collapsing the predicted step to zero (no-op) or by trivially
   matching norms (no pressure to leave init). Neither erases.

3. **The online ESD target is too high-variance to align to — and prompt variance dominates** (C,
   E). The image papers rely on short, low-diversity prompts that make the per-step ESD direction
   nearly stationary. CogVideoX's long detailed captions make it reshuffle every step; a smooth
   deterministic hypernet step cannot align to it until that variance (especially the *prompt*
   component) is averaged out.

4. **The apply path was never the problem** (D). Injecting a known-good adapter erases fire through
   our exact pipeline — isolating all failure to the learning signal and saving us from chasing
   wiring ghosts.

5. **Magnitude/target coupling is unstable** (E), and even once stabilized, **the semantic switch
   does not hold for video** (F). This is the real, still-open obstacle: a single hypernet output
   cannot stay zero on unrelated prompts while growing large on fire prompts, because CogVideoX's
   prompt embeddings for the two regimes are not cleanly separable.

**Net:** we moved from "nothing trains" (A) → "trains but to a no-op" (B) → "moves wrong
direction" (C) → "path proven, signal localized" (D) → "signal converges but diverges" (E) →
"stable, selective erasure for ~100 steps, then leaks onto unrelated" (F). The remaining blocker
is the prompt-conditioned semantic switch under CogVideoX's high-diversity text conditioning — the
one place where the image-domain assumptions break down for video.
