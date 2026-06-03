"""Architectural pieces for UnHype: a hypernetwork that emits LoRA weights for a
frozen CogVideoX transformer, and a custom Linear wrapper that accepts those
weights as plain tensors so gradients flow back into the hypernetwork."""

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Std of the final-layer weight init. Small but nonzero so the hypernet output
# actually varies with (clip embedding, step) — the prerequisite for representing
# a trajectory and for prompt conditioning. A zero weight collapses the output to
# a constant (the bias) for every input, freezing the adapter at its init.
FINAL_LAYER_WEIGHT_STD = 2e-3


@dataclass
class LoRAShape:
    path: str
    in_features: int
    out_features: int


class HyperLoRALinear(nn.Module):
    """Linear layer with an externally-provided LoRA delta.

    Unlike PEFT's LoRA, ``lora_A``/``lora_B`` are plain attributes (not
    ``nn.Parameter``s). They are assigned each step from a hypernetwork output,
    so the diffusion forward pass becomes differentiable w.r.t. the hypernetwork.
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.lora_A: torch.Tensor | None = None  # (rank, in_features)
        self.lora_B: torch.Tensor | None = None  # (out_features, rank)
        self._adapter_disabled = False

    def set_lora(self, A: torch.Tensor, B: torch.Tensor) -> None:
        self.lora_A = A
        self.lora_B = B

    def clear_lora(self) -> None:
        self.lora_A = None
        self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self._adapter_disabled or self.lora_A is None or self.lora_B is None:
            return out
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        lora_out = F.linear(F.linear(x, a), b) * self.scaling
        return out + lora_out


def _get_module(root: nn.Module, path_parts: Iterable[str]) -> nn.Module:
    obj: nn.Module = root
    for p in path_parts:
        if p.isdigit():
            obj = obj[int(p)]  # type: ignore[index]
        else:
            obj = getattr(obj, p)
    return obj


def _set_child(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    if name.isdigit():
        parent[int(name)] = new_module  # type: ignore[index]
    else:
        setattr(parent, name, new_module)


def replace_with_hyper_lora(
    transformer: nn.Module,
    target_module_names: list[str],
    rank: int,
    alpha: float,
) -> tuple[list[HyperLoRALinear], list[LoRAShape]]:
    """Replace every ``nn.Linear`` whose full dotted name ends with one of
    ``target_module_names`` with a ``HyperLoRALinear`` wrapper. Returns the new
    modules and a shape description in the order they were replaced (which is
    also the order in which the hypernet's flat output is decoded)."""
    matches: list[tuple[str, nn.Linear]] = []
    for name, module in transformer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        for tgt in target_module_names:
            if name == tgt or name.endswith("." + tgt):
                matches.append((name, module))
                break

    hyper_modules: list[HyperLoRALinear] = []
    shapes: list[LoRAShape] = []
    for name, linear in matches:
        new = HyperLoRALinear(linear, rank=rank, alpha=alpha)
        parts = name.split(".")
        parent = _get_module(transformer, parts[:-1]) if len(parts) > 1 else transformer
        _set_child(parent, parts[-1], new)
        hyper_modules.append(new)
        shapes.append(LoRAShape(path=name, in_features=linear.in_features, out_features=linear.out_features))
    return hyper_modules, shapes


@contextmanager
def disable_hyper_adapters(transformer: nn.Module):
    """Mirror of ``peft_model.disable_adapter()`` for the HyperLoRA replacement.
    Toggles a flag on each HyperLoRALinear so the LoRA branch is skipped."""
    hyper_modules = [m for m in transformer.modules() if isinstance(m, HyperLoRALinear)]
    try:
        for m in hyper_modules:
            m._adapter_disabled = True
        yield
    finally:
        for m in hyper_modules:
            m._adapter_disabled = False


def apply_hypernet_output(
    hyper_modules: list[HyperLoRALinear],
    ab_list: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for module, (A, B) in zip(hyper_modules, ab_list):
        module.set_lora(A, B)


def clear_hypernet_output(hyper_modules: list[HyperLoRALinear]) -> None:
    for m in hyper_modules:
        m.clear_lora()


def sinusoidal_step_embedding(s: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Standard diffusion-style sinusoidal embedding, used here for the
    unlearning trajectory step ``s``."""
    half = max(dim // 2, 1)
    device = s.device
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = s.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb


class Hypernetwork(nn.Module):
    """MLP that maps (CLIP embedding, unlearning step) → flat LoRA parameter vector."""

    def __init__(
        self,
        clip_dim: int,
        lora_shapes: list[LoRAShape],
        rank: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        step_embedding_dim: int = 128,
        max_step: int = 50,
    ) -> None:
        super().__init__()
        self.clip_dim = clip_dim
        self.rank = rank
        self.lora_shapes = lora_shapes
        self.step_embedding_dim = step_embedding_dim
        self.max_step = max_step

        self.output_sizes = [s.in_features * rank + rank * s.out_features for s in lora_shapes]
        self.total_output = sum(self.output_sizes)

        input_dim = clip_dim + step_embedding_dim
        layers: list[nn.Module] = []
        prev = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.SiLU())
            prev = hidden_dim
        out_layer = nn.Linear(prev, self.total_output)
        # Init so the output depends on (c, s) yet starts as an exact no-op adapter:
        #   * small nonzero weight  → θ varies with the clip embedding and step s, so
        #     θ(s+1) − θ(s) ≠ 0 and the removal loss has a real, non-degenerate target;
        #   * bias seeds standard LoRA init (A ~ kaiming, B = 0);
        #   * the weight rows that emit B are zeroed, so B = 0 *exactly* at init.
        # Hence at init: A varies with (c, s) but B ≡ 0 ⇒ A·B = 0 (base model preserved
        # on every prompt), while ∂ℒ_task/∂B ∝ A ≠ 0 and the removal gradient grows the
        # B-rows from zero, forming the trajectory. Two failure modes this avoids:
        #   - A = B = 0 (fully zero bias):   ∇_θ ℒ_task ≡ 0, nothing trains;
        #   - zero weight (constant output): adapter frozen at B = 0 with zero effect on
        #     generation, and θ(s+1) − θ(s) ≡ 0 so the zero trajectory trivially wins.
        nn.init.normal_(out_layer.weight, std=FINAL_LAYER_WEIGHT_STD)
        with torch.no_grad():
            self._zero_b_slot_rows(out_layer.weight)
            out_layer.bias.copy_(self._lora_init_bias())
        layers.append(out_layer)
        self.mlp = nn.Sequential(*layers)

    def _zero_b_slot_rows(self, weight: torch.Tensor) -> None:
        """Zero the final-layer weight rows that emit LoRA B entries, so B = 0
        exactly at init (A·B = 0, base model preserved) while A still varies with
        (c, s). The removal-loss gradient grows these rows from zero."""
        offset = 0
        for shape in self.lora_shapes:
            a_size = shape.in_features * self.rank
            b_size = self.rank * shape.out_features
            weight[offset + a_size: offset + a_size + b_size].zero_()
            offset += a_size + b_size

    def _lora_init_bias(self) -> torch.Tensor:
        """Flat bias seeding standard LoRA init (A ~ kaiming, B = 0), laid out
        exactly as ``decode`` reads it (per module: A entries then B entries)."""
        bias = torch.zeros(self.total_output)
        offset = 0
        for shape in self.lora_shapes:
            a_size = shape.in_features * self.rank
            b_size = self.rank * shape.out_features
            a = torch.empty(self.rank, shape.in_features)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            bias[offset:offset + a_size] = a.flatten()
            offset += a_size + b_size  # B-slots stay zero
        return bias

    def forward(self, clip_emb: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        step_emb = sinusoidal_step_embedding(step, self.step_embedding_dim)
        x = torch.cat([clip_emb, step_emb.to(clip_emb.dtype)], dim=-1)
        return self.mlp(x)

    def decode(self, flat: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Split a flat 1-D parameter vector into a list of (A, B) per module.

        A has shape (rank, in_features); B has shape (out_features, rank).
        Returned tensors are views into ``flat`` so autograd flows back."""
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        offset = 0
        for shape in self.lora_shapes:
            a_size = shape.in_features * self.rank
            b_size = self.rank * shape.out_features
            A = flat[offset:offset + a_size].view(self.rank, shape.in_features)
            offset += a_size
            B = flat[offset:offset + b_size].view(shape.out_features, self.rank)
            offset += b_size
            out.append((A, B))
        return out
