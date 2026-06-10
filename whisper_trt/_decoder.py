# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""KV-cached Whisper text-decoder modules and their TensorRT runtime wrapper.

Split out of ``model.py`` to keep that module focused on the builder/loader
plumbing. These classes implement the autoregressive decode optimization:
cross-attention K/V are projected once per utterance and self-attention keeps
a growing KV cache, so per-token work is O(prefix) instead of O(prefix^2).
"""

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from whisper.model import LayerNorm, ModelDimensions, Tensor


@dataclass
class TextDecoderState:
    """Non-engine state the runtime decoder needs alongside its TRT engines."""

    token_embedding: nn.Embedding
    positional_embedding: torch.Tensor
    ln: LayerNorm
    mask: torch.Tensor


@dataclass
class DecodeRequest:
    """Inputs required for one autoregressive decode pass."""

    out_tokens: torch.Tensor
    cur_len: int
    prompt_len: int
    max_len: int
    audio_features: Tensor
    stream: bool


@dataclass
class DecoderEngines:
    """The three TensorRT engines backing the KV-cached decoder."""

    cross_kv: Any
    prefill: Any
    step: Any


def _split_heads(x: Tensor, n_head: int) -> Tensor:
    """Reshape [batch, seq, n_state] into [batch, n_head, seq, head_dim]."""
    batch, seq, n_state = x.shape
    return x.view(batch, seq, n_head, n_state // n_head).permute(0, 2, 1, 3)


def _attention(
    q: Tensor, k: Tensor, v: Tensor, n_head: int, mask: Tensor | None = None
) -> Tensor:
    """Scaled-dot-product attention returning [batch, seq, n_state].

    ``mask`` (additive, ``[n_q, n_k]``) is applied to the scores before
    softmax for the multi-token causal prefill pass. Single-token decode
    steps pass ``mask=None`` — the lone query attends to every cached key,
    causal by construction. Uses the symmetric ``scale ** 0.25`` split for
    parity with whisper's manual-attention path.
    """
    head_dim = q.shape[-1] // n_head
    scale = head_dim**-0.25
    qh = _split_heads(q, n_head) * scale
    kh = _split_heads(k, n_head) * scale
    vh = _split_heads(v, n_head)
    scores = qh @ kh.transpose(-1, -2)
    if mask is not None:
        scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    out = (weights @ vh).permute(0, 2, 1, 3)
    return out.flatten(start_dim=2)


class CrossKVProjector(nn.Module):
    """Project encoder features into cross-attention K/V once.

    Cross-attention keys/values depend only on the (fixed) encoder output, so
    computing them a single time per utterance avoids re-projecting all
    ~1500 audio frames on every decode step. K and V are stacked along a
    leading axis (``[2, n_layers, 1, n_audio_ctx, n_state]``) so the engine
    has a single output.
    """

    def __init__(self, blocks: Any) -> None:
        super().__init__()
        self.blocks = blocks

    @torch.no_grad()
    def forward(self, xa: Tensor) -> Tensor:
        """Return stacked cross K/V, ``[2, n_layers, 1, n_audio_ctx, n_state]``."""
        keys: list[Tensor] = []
        values: list[Tensor] = []
        for block in cast(list[Any], self.blocks):
            keys.append(block.cross_attn.key(xa))
            values.append(block.cross_attn.value(xa))
        return torch.stack([torch.stack(keys, 0), torch.stack(values, 0)], 0)

    def summary(self) -> str:
        """Return a short human-readable component summary."""
        return "Cross-attention K/V projector"


class CachedDecoderStep(nn.Module):
    """Single decode step with self-attention KV cache + precomputed cross K/V.

    Self-attention recomputes only the new token's key/value and concatenates
    them onto the running cache, so per-step work is O(prefix) instead of the
    O(prefix^2) of re-running the whole sequence each token. The causal mask
    is unnecessary: the lone query legitimately attends to every cached
    (past) key. K/V pairs travel stacked on a leading axis to keep the engine
    I/O to two cache tensors in and one out.
    """

    def __init__(self, blocks: Any, n_head: int) -> None:
        super().__init__()
        self.blocks = blocks
        self.n_head = n_head

    def _block_step(
        self, block: Any, x: Tensor, self_kv_i: Tensor, cross_kv_i: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one residual block, returning hidden state and grown self K/V.

        ``self_kv_i``/``cross_kv_i`` are this layer's ``[2, 1, seq, n_state]``
        K/V slices (index 0 = key, 1 = value).
        """
        attended = block.attn_ln(x)
        k = torch.cat([self_kv_i[0], block.attn.key(attended)], dim=1)
        v = torch.cat([self_kv_i[1], block.attn.value(attended)], dim=1)
        q = block.attn.query(attended)
        x = x + block.attn.out(_attention(q, k, v, self.n_head))

        qc = block.cross_attn.query(block.cross_attn_ln(x))
        x = x + block.cross_attn.out(
            _attention(qc, cross_kv_i[0], cross_kv_i[1], self.n_head)
        )

        x = x + block.mlp(block.mlp_ln(x))
        return x, k, v

    @torch.no_grad()
    def forward(
        self, x: Tensor, self_kv: Tensor, cross_kv: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Advance one token.

        Args:
            x: New token hidden state, ``[1, 1, n_state]``.
            self_kv: Cached self-attention K/V, ``[2, n_layers, 1, past, n_state]``.
            cross_kv: Precomputed cross K/V, ``[2, n_layers, 1, n_audio_ctx, n_state]``.

        Returns:
            The final hidden state ``[1, 1, n_state]`` and the grown
            self-attention cache ``[2, n_layers, 1, past + 1, n_state]``.
        """
        new_k: list[Tensor] = []
        new_v: list[Tensor] = []
        for i, block in enumerate(cast(list[Any], self.blocks)):
            x, k, v = self._block_step(block, x, self_kv[:, i], cross_kv[:, i])
            new_k.append(k)
            new_v.append(v)
        return x, torch.stack([torch.stack(new_k, 0), torch.stack(new_v, 0)], 0)

    def summary(self) -> str:
        """Return a short human-readable component summary."""
        return "Cached single-token decoder step"


class PrefillDecoder(nn.Module):
    """Process the whole prompt in one masked pass, emitting the initial cache.

    Runs full causal self-attention over the prompt (length >= 1) so the
    step engine never receives an empty cache — TensorRT cannot bind a
    zero-length input tensor. Returns the last position's hidden state (which
    predicts the first generated token) and the prompt's self-attention K/V.
    """

    def __init__(self, blocks: Any, n_head: int) -> None:
        super().__init__()
        self.blocks = blocks
        self.n_head = n_head

    def _block(
        self, block: Any, x: Tensor, mask: Tensor, cross_kv_i: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one residual block over the full prompt, returning self K/V."""
        attended = block.attn_ln(x)
        k = block.attn.key(attended)
        v = block.attn.value(attended)
        q = block.attn.query(attended)
        x = x + block.attn.out(_attention(q, k, v, self.n_head, mask))

        qc = block.cross_attn.query(block.cross_attn_ln(x))
        x = x + block.cross_attn.out(
            _attention(qc, cross_kv_i[0], cross_kv_i[1], self.n_head)
        )

        x = x + block.mlp(block.mlp_ln(x))
        return x, k, v

    @torch.no_grad()
    def forward(
        self, x: Tensor, cross_kv: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Prefill the cache from the prompt.

        Args:
            x: Prompt hidden states, ``[1, prompt_len, n_state]``.
            cross_kv: Precomputed cross K/V, ``[2, n_layers, 1, n_audio_ctx, n_state]``.
            mask: Additive causal mask, ``[prompt_len, prompt_len]``.

        Returns:
            The final position's hidden state ``[1, 1, n_state]`` and the
            prompt self-attention cache ``[2, n_layers, 1, prompt_len, n_state]``.
        """
        new_k: list[Tensor] = []
        new_v: list[Tensor] = []
        for i, block in enumerate(cast(list[Any], self.blocks)):
            x, k, v = self._block(block, x, mask, cross_kv[:, i])
            new_k.append(k)
            new_v.append(v)
        self_kv = torch.stack([torch.stack(new_k, 0), torch.stack(new_v, 0)], 0)
        return x[:, -1:, :], self_kv

    def summary(self) -> str:
        """Return a short human-readable component summary."""
        return "Prompt prefill decoder"


class TextDecoderTRTKV(nn.Module):
    """KV-cached Whisper text decoder backed by three TensorRT engines."""

    def __init__(
        self,
        engines: DecoderEngines,
        state: TextDecoderState,
        dims: ModelDimensions,
    ) -> None:
        super().__init__()
        self.engines = engines
        self.token_embedding = state.token_embedding
        self.positional_embedding = state.positional_embedding
        self.ln = state.ln
        self.register_buffer("mask", state.mask, persistent=False)
        self.n_layers = dims.n_text_layer
        self.n_state = dims.n_text_state

    @torch.no_grad()
    def compute_cross_kv(self, xa: Tensor) -> Tensor:
        """Project encoder features into cached cross K/V (once per utterance)."""
        return self.engines.cross_kv(xa)

    @torch.no_grad()
    def prefill(self, prompt_ids: list[int], cross_kv: Tensor) -> tuple[Tensor, Tensor]:
        """Prime the KV cache from the prompt in a single masked pass.

        Returns the logits predicting the first generated token and the
        initial self-attention cache (length == ``len(prompt_ids)``).
        """
        device = cross_kv.device
        prompt_len = len(prompt_ids)
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        pos = self.positional_embedding[:prompt_len].to(device)
        hidden_in = self.token_embedding(tokens).to(device) + pos
        mask = cast(Tensor, self.mask)[:prompt_len, :prompt_len].to(device)
        last_hidden, self_kv = self.engines.prefill(hidden_in, cross_kv, mask)
        last_hidden = self.ln(last_hidden)
        weight = self.token_embedding.weight.to(device)
        logits = (last_hidden @ torch.transpose(weight, 0, 1)).float()
        return logits, self_kv

    @torch.no_grad()
    def step(
        self,
        token_id: int,
        position: int,
        self_kv: Tensor,
        cross_kv: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Decode one token, returning final-position logits and the grown cache."""
        device = cross_kv.device
        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        pos = self.positional_embedding[position : position + 1].to(device)
        hidden_in = self.token_embedding(token).to(device) + pos
        hidden, new_self_kv = self.engines.step(hidden_in, self_kv, cross_kv)
        hidden = self.ln(hidden)[:, -1:, :]
        weight = self.token_embedding.weight.to(device)
        logits = (hidden @ torch.transpose(weight, 0, 1)).float()
        return logits, new_self_kv

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Unused. Drive the decoder via compute_cross_kv() and step()."""
        raise NotImplementedError(
            "TextDecoderTRTKV has no forward(); use compute_cross_kv()/step()."
        )

    def summary(self) -> str:
        """Return a short human-readable component summary."""
        return "TensorRT KV-cached text decoder wrapper"
