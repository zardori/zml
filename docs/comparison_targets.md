# Comparison Targets: Which Concepts to Attack Next

We developed frame_replace on **fire**, which is a convenient research concept but one nobody else
reports. To position the method against published work we need to erase concepts that other T2V
unlearning papers also erase, measured with the same instruments. This document collects what those
papers do and recommends the order in which we should follow.

---

## 1. What other papers evaluate

| Paper | Base models | Concepts | Detectors / metrics |
|---|---|---|---|
| [T2VUnlearning (2505.17550)](https://arxiv.org/abs/2505.17550) | CogVideoX-2B/5B, HunyuanVideo | nudity; 5 celebrity identities; 10 ImageNet objects | NudeNet nudity rate; ArcFace ID-similarity; per-frame classifier ESR/PSR (classifier unnamed — we use ResNet-50, see [`imagenet_objects.md`](imagenet_objects.md)); VBench |
| [VideoEraser (2508.15314)](https://arxiv.org/abs/2508.15314) | AnimateDiff, LaVie, ZeroScope, ModelScope, CogVideoX | Imagenette objects; 5 artists (Van Gogh, Picasso, …); 5 celebrities; toxic categories (violence, pornography, …) | ResNet-50 ACCe/ACCu; GIPHY celebrity detector; DOVER; attack success rate vs Ring-A-Bell / MMA-Diffusion / UnlearnDiffAtk |
| [Video Unlearning via Low-Rank Refusal Vector (2506.07891)](https://arxiv.org/abs/2506.07891) | Open-Sora, ZeroScope | T2VSafetyBench / SafeSora categories | benchmark-native safety scores |

T2VUnlearning is the closest comparison: same base model family (CogVideoX-5b) and same
v-prediction parameterization.

**Prompt sets / benchmarks worth adopting** so numbers are directly comparable: **SafeSora** and
**Ring-A-Bell** (nudity), **Imagenette / ImageNet** class prompts (objects), **VBench** (utility /
collateral damage).

## 2. Recommended order

### 2.1 Nudity — in progress

Reported by all three papers, with an objective off-the-shelf detector (NudeNet) that we already
wrap in `zml/benchmarks/check_for_nudity.py`. The partial-clip blocker is solved by split-prompt
(see [`split_prompt.md`](split_prompt.md)); exp061 built the dataset, exp062 is the training pilot.

Remaining work for a publishable comparison:

- evaluate on SafeSora / Ring-A-Bell style prompts, not only our own written ones;
- build a nudity **`related`** preservation set (e.g. swimwear, partially clothed, anatomy in art) —
  currently missing, `control_related_prompts` is a required-but-unused slot in exp062;
- replace the fire-era exp041 retention anchors with a nudity-appropriate preservation set.

### 2.2 ImageNet / Imagenette objects — in progress

The best second axis, for four reasons:

1. **Most overlap** — both T2VUnlearning (10 ImageNet classes) and VideoEraser (Imagenette) report
   it, so one concept buys two comparison points.
2. **Free instrumentation** — the detector is an off-the-shelf image classifier applied per frame;
   no bespoke detector engineering, unlike nudity or style.
3. **Unambiguous metric** — ESR/PSR (erasure/preservation success rate) via top-k accuracy, no
   judgement calls.
4. **Native regime for frame_replace** — an object is *spatially and temporally localized*, which is
   precisely what the frame-local edit was designed for.

The protocol is implemented: a per-frame ResNet-50 (`zml/benchmarks/check_for_object.py`), an ESR/PSR
eval mode (`mode: imagenet`), the ten-class prompt sets, and a two-class pilot (chain saw and church)
in exp064–exp072. Full write-up — class list with ImageNet indices, exact metric definitions, our
deviations from the papers, threshold calibration, and status: **[`imagenet_objects.md`](imagenet_objects.md)**.

### 2.3 Celebrity identity — if time allows

Strong comparison point (T2VUnlearning and VideoEraser both use 5 identities) with an objective
metric (ArcFace ID-similarity, GIPHY celebrity detector). But identity is present in every frame,
same situation as nudity, so it depends entirely on split-prompt working. Worth doing only after
2.2 confirms the pipeline generalizes.

### 2.4 Artistic style — weak fit, deprioritize

Style is a *global* property of every pixel of every frame, so there is nothing frame-local to
replace, and there is no crisp detector (the papers fall back to GPT-4o as a judge). Both the method
and the evaluation would be fighting us.

## 3. Robustness, eventually

VideoEraser reports attack success rate against Ring-A-Bell, MMA-Diffusion, P4D and UnlearnDiffAtk.
Reviewers of erasure papers now expect some red-teaming; worth budgeting once erasure itself holds
on two concepts.
