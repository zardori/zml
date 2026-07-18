"""Mask-based localization regularizer for ESD-style concept erasure.

Implements the localization loss `L_loc` from T2VUnlearning (arXiv:2505.17550, Eq. 9):

    L_loc = (1 / L) * sum_l || o^l ⊙ (1 - M) ||^2

where `o^l` is the erasing adapter's (LoRA) output at layer `l` and `M` is a spatial-temporal
concept mask in [0, 1] over the video tokens. The term drives the LoRA edit to zero *outside* the
region where the target concept appears, keeping erasure localized so unrelated content is
preserved.

CogVideoX attention is a *joint* attention over `concat([text_tokens, video_tokens])`; the
text->video block of the query-key map behaves like cross-attention. The mask is built by taking,
for every video query, its soft attention over the concept word's text key(s), averaged over heads
and attention blocks (see `MaskCaptureProcessor` / `LocalizationHelper.build_mask`).

The faithful analog of the paper's adapter output `o^l` here is the LoRA delta at each block's
attention output projection (`attn1.to_out.0`), restricted to the video tokens.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb


# `mask_threshold` <= this keeps the (min-max normalized) soft mask; above it, binarize.
DEFAULT_MASK_THRESHOLD = 0.0
# Numerical floor for the min-max normalization denominator.
MASK_EPS = 1e-6


def find_concept_token_indices(
    tokenizer,
    prompt: str,
    concept_word: str,
    max_len: int = 226,
) -> list[int]:
    """Return the padded-sequence positions of `concept_word`'s tokens inside `prompt`.

    Matches the token-id subsequence of `concept_word` (tokenized on its own, minus trailing EOS)
    against the prompt's padded `input_ids`. Returns an empty list if the word is absent, in which
    case the caller should fall back to all non-padding text positions.
    """
    prompt_ids: list[int] = tokenizer(
        prompt, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt"
    ).input_ids[0].tolist()

    # Tokenize the concept word alone; drop special/EOS tokens so we match its content ids only.
    special_ids = set(tokenizer.all_special_ids)
    word_ids = [i for i in tokenizer(concept_word, add_special_tokens=False).input_ids if i not in special_ids]
    if not word_ids:
        return []

    positions: list[int] = []
    for start in range(len(prompt_ids) - len(word_ids) + 1):
        if prompt_ids[start : start + len(word_ids)] == word_ids:
            positions.extend(range(start, start + len(word_ids)))
    return positions


def non_padding_token_indices(tokenizer, prompt: str, max_len: int = 226) -> list[int]:
    """Positions of the real (non-pad) text tokens in the padded sequence.

    Fallback mask source when the concept word is not found in a prompt.
    """
    ids = tokenizer(
        prompt, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt"
    ).input_ids[0]
    pad_id = tokenizer.pad_token_id
    return (ids != pad_id).nonzero(as_tuple=True)[0].tolist()


class MaskCaptureProcessor:
    """Drop-in mirror of ``CogVideoXAttnProcessor2_0`` that also records a text->video mask.

    The attention output is computed identically to the stock processor (so wrapping a forward pass
    leaves its result numerically unchanged), and additionally, for the video queries, the softmax
    attention over the *text* keys is reduced to a per-video-token score at the concept columns.
    Scores are appended to ``collector`` (one ``[num_video_tokens]`` tensor per attention block).
    """

    def __init__(self, collector: list[torch.Tensor], concept_token_indices: list[int]) -> None:
        self.collector = collector
        self.concept_token_indices = concept_token_indices

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # --- mask capture: video queries attending to text keys (cross-attention-like block) ---
        self.collector.append(
            self._text_to_video_scores(query, key, text_seq_length, head_dim)
        )

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states

    def _text_to_video_scores(
        self, query: torch.Tensor, key: torch.Tensor, text_seq_length: int, head_dim: int
    ) -> torch.Tensor:
        """Per-video-token attention mass on the concept text tokens, averaged over heads.

        Returns a ``[num_video_tokens]`` tensor for a single (batch-size-1) forward pass.
        """
        q_video = query[:, :, text_seq_length:]  # [B, heads, V, d]
        k_text = key[:, :, :text_seq_length]  # [B, heads, T, d]
        logits = torch.matmul(q_video, k_text.transpose(-1, -2)) / math.sqrt(head_dim)  # [B, heads, V, T]
        probs = logits.float().softmax(dim=-1)  # distribution over text keys per video query

        cols = self.concept_token_indices if self.concept_token_indices else list(range(text_seq_length))
        concept_mass = probs[..., cols].sum(dim=-1)  # [B, heads, V]
        return concept_mass.mean(dim=1)[0].detach()  # mean over heads, batch 0 -> [V]


@dataclass
class LocalizationHelper:
    """Wiring for building the concept mask and the localization loss on a PEFT transformer."""

    transformer: torch.nn.Module
    grid_shape: tuple[int, int, int]  # (F', H', W') video-token grid
    text_seq_len: int
    threshold: float = DEFAULT_MASK_THRESHOLD

    def __post_init__(self) -> None:
        # Attention blocks whose text->video map feeds the mask, and their output-projection LoRA.
        self._attn_modules = [
            m for name, m in self.transformer.named_modules() if name.endswith("attn1")
        ]
        self._out_proj_loras = [
            m for name, m in self.transformer.named_modules() if name.endswith("attn1.to_out.0")
        ]
        if not self._attn_modules or not self._out_proj_loras:
            raise RuntimeError("Could not locate attn1 / to_out.0 modules for localization.")

    @staticmethod
    def compute_grid_shape(
        transformer, num_frames: int, height: int, width: int
    ) -> tuple[int, int, int]:
        """Video-token grid (F', H', W') after patch embedding, from the transformer config."""
        patch_size = transformer.config.patch_size
        patch_size_t = getattr(transformer.config, "patch_size_t", None)
        f = num_frames if not patch_size_t else num_frames // patch_size_t
        return (f, height // patch_size, width // patch_size)

    @property
    def num_video_tokens(self) -> int:
        f, h, w = self.grid_shape
        return f * h * w

    @contextmanager
    def build_mask(self, concept_token_indices: list[int]) -> Iterator[list]:
        """Temporarily install capture processors on all attn1 blocks.

        Yields a one-element list; after the wrapped (teacher) forward pass completes, the built
        mask ``M`` (shape ``grid_shape``, values in ``[0, 1]``, detached) is stored at index 0.
        """
        collector: list[torch.Tensor] = []
        originals = [m.get_processor() for m in self._attn_modules]
        for m in self._attn_modules:
            m.set_processor(MaskCaptureProcessor(collector, concept_token_indices))
        out: list = [None]
        try:
            yield out
            out[0] = self._reduce_mask(collector)
        finally:
            for m, proc in zip(self._attn_modules, originals):
                m.set_processor(proc)

    def _reduce_mask(self, collector: list[torch.Tensor]) -> torch.Tensor:
        """Average per-block scores, min-max normalize, reshape to the grid, optionally binarize."""
        stacked = torch.stack(collector, dim=0).mean(dim=0)  # [V]
        lo, hi = stacked.min(), stacked.max()
        norm = (stacked - lo) / (hi - lo + MASK_EPS)
        if self.threshold > 0.0:
            norm = (norm >= self.threshold).to(norm.dtype)
        return norm.reshape(self.grid_shape).detach()

    @contextmanager
    def capture_adapter_inputs(self) -> Iterator[list[tuple]]:
        """Register pre-hooks capturing each ``to_out.0`` LoRA's *input* (video tokens, detached).

        The localization delta ``o^l = scaling * B(A(x))`` is recomputed afterwards from these
        inputs (see ``localization_loss``) rather than captured during the forward. Detaching ``x``
        keeps the enormous base forward graph out of the localization loss — grad then flows only
        through the LoRA ``A``/``B`` weights, which is exactly what ``L_loc`` regularizes — so
        gradient checkpointing can stay *enabled* for the student forward. Capturing the ~rank-r
        delta output live instead would force the whole forward graph to be retained (OOM).

        Yields a list holding one ``(lora_module, x_video)`` pair per block after the forward.
        """
        captured: list[tuple] = []
        handles = []
        for lora in self._out_proj_loras:
            handles.append(lora.register_forward_pre_hook(
                _make_input_hook(captured, self.text_seq_len)
            ))
        try:
            yield captured
        finally:
            for h in handles:
                h.remove()

    def localization_loss(self, mask: torch.Tensor, captured: list[tuple]) -> torch.Tensor:
        """`mean_l mean_pos || o^l_video ⊙ (1 - M) ||^2`, with M detached and broadcast over dim.

        Recomputes each block's LoRA delta ``o^l`` from its detached input, so the loss carries a
        graph over only the (tiny, rank-r) adapter recompute — not the base transformer forward.
        """
        keep = (1.0 - mask).reshape(self.num_video_tokens, 1).float()  # [V, 1]
        per_layer = [
            (_lora_delta(lora, x_video).float() * keep).pow(2).sum(dim=-1).mean()
            for lora, x_video in captured
        ]
        return torch.stack(per_layer).mean()


def _active_adapter(lora_module) -> str:
    active = getattr(lora_module, "active_adapters", None)
    if active:
        return active[0]
    return next(iter(lora_module.lora_B.keys()))


def _make_input_hook(sink: list[tuple], text_seq_len: int):
    """Forward pre-hook on a ``to_out.0`` LoRA layer capturing its detached input on video tokens."""

    def hook(module, args: tuple) -> None:
        # args[0]: [B, seq, dim_in]; keep video tokens of the single-sample batch, detach so the
        # base forward graph is not retained through the localization loss.
        sink.append((module, args[0][0, text_seq_len:].detach()))

    return hook


def _lora_delta(lora_module, x_video: torch.Tensor) -> torch.Tensor:
    """Recompute the scaled LoRA output ``scaling * B(A(dropout(x)))`` for the given input.

    Mirrors PEFT's ``lora.Linear`` update. With ``x_video`` detached, the resulting delta's graph
    reaches only the adapter weights, keeping the localization loss cheap.
    """
    adapter = _active_adapter(lora_module)
    lora_a = lora_module.lora_A[adapter]
    lora_b = lora_module.lora_B[adapter]
    dropout = lora_module.lora_dropout[adapter]
    scaling = lora_module.scaling[adapter]
    return lora_b(lora_a(dropout(x_video))) * scaling
