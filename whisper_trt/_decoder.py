# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""KV-cached Whisper text-decoder modules and their TensorRT runtime wrapper.

Split out of ``model.py`` to keep that module focused on the builder/loader
plumbing. These classes implement the autoregressive decode optimization:
cross-attention K/V are projected once per utterance and self-attention keeps
a growing KV cache, so per-token work is O(prefix) instead of O(prefix^2).
"""

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from whisper.model import LayerNorm, ModelDimensions, Tensor

logger = logging.getLogger(__name__)


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
    # When set, decode returns empty text if the probability of the
    # ``<|nospeech|>`` token at the first decode position meets this threshold.
    # ``None`` disables the check. See ``WhisperTRT._is_no_speech``.
    no_speech_threshold: float | None = None


@dataclass
class DecoderEngines:
    """The three TensorRT engines backing the KV-cached decoder."""

    cross_kv: Any
    prefill: Any
    step: Any


def _engine_torch_dtype(engine: Any, name: str) -> torch.dtype:
    """Return the torch dtype TensorRT expects for one engine binding."""
    import numpy as np
    import tensorrt as trt

    return torch.from_numpy(
        np.empty(0, dtype=trt.nptype(engine.get_tensor_dtype(name)))
    ).dtype


def _bind_tensor(context: Any, name: str, tensor: Tensor, shape: bool = False) -> None:
    """Point one engine binding at a tensor, checking the status TRT returns.

    These setters return a bool rather than raising; ignoring it is how a
    rejected shape or address turns into silent corruption (see the same
    reasoning in torch2trt's TRTModule).
    """
    if shape and not context.set_input_shape(name, tuple(tensor.shape)):
        raise RuntimeError(
            f"TensorRT rejected shape {tuple(tensor.shape)} for '{name}'."
        )
    if not context.set_tensor_address(name, tensor.data_ptr()):
        raise RuntimeError(f"TensorRT rejected the address for '{name}'.")


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


class GraphCachedDecoderStep(nn.Module):
    """Single decode step over a FIXED-CAPACITY self-attention cache.

    Same arithmetic as :class:`CachedDecoderStep`, restructured so that every
    tensor entering and leaving the engine has a shape that never changes:

    * the cache arrives at its full capacity ``C`` (not ``past``), with the
      unwritten tail masked out rather than absent, and
    * only the new token's K/V leaves the engine (``[2, L, 1, 1, S]``), instead
      of the whole grown cache. The caller scatters that slice into its own
      cache buffer.

    Constant shapes are the entire point: a CUDA graph bakes in both shapes and
    addresses, so the growing ``[2, L, 1, past, S]`` cache of the dynamic
    decoder forces a re-capture per token, which costs more than it saves.
    With this form one capture replays for every token of every utterance.

    ``mask`` is additive over ``C + 1`` keys -- the ``C`` cache slots plus the
    new token appended at the end -- so the caller controls which slots are
    live without changing any shape.
    """

    def __init__(self, blocks: Any, n_head: int) -> None:
        super().__init__()
        self.blocks = blocks
        self.n_head = n_head

    def _block_step(
        self, block: Any, x: Tensor, self_kv_i: Tensor, cross_kv_i: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one residual block, returning the hidden state and new K/V."""
        attended = block.attn_ln(x)
        k_new = block.attn.key(attended)
        v_new = block.attn.value(attended)
        q = block.attn.query(attended)
        # The new entry is appended rather than scattered in: TensorRT cannot
        # alias an input to an output, so the cache is read-only here and the
        # caller writes k_new/v_new into it after the engine returns.
        k = torch.cat([self_kv_i[0], k_new], dim=1)
        v = torch.cat([self_kv_i[1], v_new], dim=1)
        x = x + block.attn.out(_attention(q, k, v, self.n_head, mask))

        qc = block.cross_attn.query(block.cross_attn_ln(x))
        x = x + block.cross_attn.out(
            _attention(qc, cross_kv_i[0], cross_kv_i[1], self.n_head)
        )

        x = x + block.mlp(block.mlp_ln(x))
        return x, k_new, v_new

    @torch.no_grad()
    def forward(
        self, x: Tensor, self_kv: Tensor, cross_kv: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Advance one token against a fixed-capacity cache.

        Args:
            x: New token hidden state, ``[1, 1, n_state]``.
            self_kv: Full-capacity cache, ``[2, n_layers, 1, C, n_state]``.
            cross_kv: Precomputed cross K/V, ``[2, n_layers, 1, n_audio_ctx, n_state]``.
            mask: Additive mask over ``[1, C + 1]`` keys; 0 for live slots and
                the new token, ``-inf`` for slots not yet written.

        Returns:
            The final hidden state ``[1, 1, n_state]`` and only this token's
            new K/V, ``[2, n_layers, 1, 1, n_state]``.
        """
        new_k: list[Tensor] = []
        new_v: list[Tensor] = []
        for i, block in enumerate(cast(list[Any], self.blocks)):
            x, k, v = self._block_step(block, x, self_kv[:, i], cross_kv[:, i], mask)
            new_k.append(k)
            new_v.append(v)
        return x, torch.stack([torch.stack(new_k, 0), torch.stack(new_v, 0)], 0)

    def summary(self) -> str:
        """Return a short human-readable component summary."""
        return "Fixed-capacity cached decoder step (CUDA-graph friendly)"


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
        logits = (last_hidden.to(weight.dtype) @ torch.transpose(weight, 0, 1)).float()
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
        logits = (hidden.to(weight.dtype) @ torch.transpose(weight, 0, 1)).float()
        return logits, new_self_kv

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Unused. Drive the decoder via compute_cross_kv() and step()."""
        raise NotImplementedError(
            "TextDecoderTRTKV has no forward(); use compute_cross_kv()/step()."
        )

    def summary(self) -> str:
        """Return a short human-readable component summary."""
        return "TensorRT KV-cached text decoder wrapper"


class TextDecoderTRTKVGraph(nn.Module):
    """KV-cached decoder whose per-token step runs from a captured CUDA graph.

    Shares the cross-K/V and prefill engines with :class:`TextDecoderTRTKV`;
    only the per-token step differs. Every buffer the step touches is allocated
    once and never moves, so the whole chain -- token embedding, positional
    embedding, the TensorRT enqueue, the cache scatter, the output projection
    and the argmax -- is captured into a single CUDA graph and replayed per
    token.

    The measured problem this solves: the per-token loop is host-launch-bound,
    not compute-bound. Issuing the work costs about as much wall time as the
    work itself (measured: 2.951 ms of a 2.960 ms step on small.en), and graph
    replay reissues the identical launches for ~6 us.

    Token feedback stays on the device: the argmax result is written straight
    back into the token buffer the next replay reads, so no host round-trip is
    needed to advance. One ``.item()`` per token remains, purely to test for
    end-of-transcript, which preserves the dynamic decoder's exact stopping
    behaviour.
    """

    def __init__(
        self,
        engines: DecoderEngines,
        state: TextDecoderState,
        dims: ModelDimensions,
        capacity: int | None = None,
    ) -> None:
        super().__init__()
        self.engines = engines
        self.token_embedding = state.token_embedding
        self.positional_embedding = state.positional_embedding
        self.ln = state.ln
        self.register_buffer("mask", state.mask, persistent=False)
        self.n_layers = dims.n_text_layer
        self.n_state = dims.n_text_state
        self.n_audio_ctx = dims.n_audio_ctx
        # Capacity is the hard ceiling on generated length; n_text_ctx matches
        # what the dynamic decoder could reach, so behaviour is unchanged.
        self.capacity = int(capacity or dims.n_text_ctx)
        self._graph: Any = None
        self._capture_failed = False
        self._ready = False
        self._host_pos = 0

    # -- setup ---------------------------------------------------------------
    def _allocate(self, device: torch.device) -> None:
        """Allocate every static buffer and bind the engine to them once."""
        engine = self.engines.step.engine
        cap = self.capacity
        n_l, n_s = self.n_layers, self.n_state

        def dt(name: str) -> torch.dtype:
            return _engine_torch_dtype(engine, name)

        self._x = torch.zeros(1, 1, n_s, dtype=dt("x"), device=device)
        self._cache = torch.zeros(
            2, n_l, 1, cap, n_s, dtype=dt("self_kv"), device=device
        )
        self._cross = torch.zeros(
            2, n_l, 1, self.n_audio_ctx, n_s, dtype=dt("cross_kv"), device=device
        )
        self._mask_row = torch.zeros(1, cap + 1, dtype=dt("mask"), device=device)
        self._hidden = torch.zeros(1, 1, n_s, dtype=dt("hidden"), device=device)
        self._new_kv = torch.zeros(
            2, n_l, 1, 1, n_s, dtype=dt("new_self_kv"), device=device
        )

        # Device-resident loop state. These must be tensors, not Python ints:
        # a captured graph replays fixed instructions, so the step position and
        # the current token have to be values the kernels *read* at replay time
        # rather than constants baked in at capture time.
        self._pos = torch.zeros(1, dtype=torch.int64, device=device)
        self._token = torch.zeros(1, 1, dtype=torch.int64, device=device)

        # Row p is the additive mask for a token at absolute position p: cache
        # slots 0..p-1 are live, slots p..C-1 are not yet written, and the
        # appended new token (index C) is always live.
        table = torch.full((cap, cap + 1), float("-inf"))
        for p in range(cap):
            table[p, :p] = 0.0
        table[:, cap] = 0.0
        self._mask_table = table.to(device=device, dtype=self._mask_row.dtype)

        weight = self.token_embedding.weight.to(device)
        self._weight_t = torch.transpose(weight, 0, 1).contiguous()

        context = self.engines.step.context
        _bind_tensor(context, "x", self._x, shape=True)
        _bind_tensor(context, "self_kv", self._cache, shape=True)
        _bind_tensor(context, "cross_kv", self._cross, shape=True)
        _bind_tensor(context, "mask", self._mask_row, shape=True)
        _bind_tensor(context, "hidden", self._hidden)
        _bind_tensor(context, "new_self_kv", self._new_kv)
        self._ready = True

    def _step_body(self) -> None:
        """One decode step, entirely in-place on the static buffers."""
        torch.index_select(self._mask_table, 0, self._pos, out=self._mask_row)
        emb = self.token_embedding(self._token)
        pos_emb = self.positional_embedding.index_select(0, self._pos)
        self._x.copy_(emb + pos_emb)

        stream = torch.cuda.current_stream().cuda_stream
        if not self.engines.step.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT failed to enqueue the decoder step engine.")

        # Commit this token's K/V into the cache at its own position. Scattering
        # by a device-side index is what lets one captured graph serve every
        # position -- a Python slice would bake the offset in at capture time.
        self._cache.index_copy_(3, self._pos, self._new_kv)

        hidden = self.ln(self._hidden)
        logits = hidden.to(self._weight_t.dtype) @ self._weight_t
        self._token.copy_(logits.argmax(dim=-1))
        self._pos.add_(1)

    def _capture(self) -> None:
        """Warm up on a side stream, then capture one step."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._step_body()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step_body()
        torch.cuda.synchronize()
        self._graph = graph

    # -- per-utterance API ---------------------------------------------------
    @torch.no_grad()
    def compute_cross_kv(self, xa: Tensor) -> Tensor:
        """Project encoder features into cached cross K/V (once per utterance)."""
        return self.engines.cross_kv(xa)

    @torch.no_grad()
    def prefill(self, prompt_ids: list[int], cross_kv: Tensor) -> tuple[Tensor, Tensor]:
        """Prime the KV cache from the prompt in a single masked pass."""
        device = cross_kv.device
        prompt_len = len(prompt_ids)
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        pos = self.positional_embedding[:prompt_len].to(device)
        hidden_in = self.token_embedding(tokens).to(device) + pos
        mask = cast(Tensor, self.mask)[:prompt_len, :prompt_len].to(device)
        last_hidden, self_kv = self.engines.prefill(hidden_in, cross_kv, mask)
        last_hidden = self.ln(last_hidden)
        weight = self.token_embedding.weight.to(device)
        logits = (last_hidden.to(weight.dtype) @ torch.transpose(weight, 0, 1)).float()
        return logits, self_kv

    @torch.no_grad()
    def begin(
        self, cross_kv: Tensor, prompt_kv: Tensor, prompt_len: int, first_token: int
    ) -> None:
        """Reset the static buffers for a new utterance.

        Capturing happens here, on first use, because the graph must be
        recorded against buffers that already exist and are already bound.
        """
        device = cross_kv.device
        if not self._ready:
            self._allocate(device)
        if self._graph is None and not self._capture_failed:
            # Capture runs the body a few times and therefore dirties the loop
            # state; everything it touches is reset immediately below.
            try:
                self._capture()
            except Exception as err:
                # Graph capture is on by default, so a driver/TRT combination
                # that refuses to capture must degrade to plain enqueues rather
                # than break transcription. The fixed-capacity engine runs
                # correctly either way; only the launch-overhead saving is lost.
                self._capture_failed = True
                logger.warning(
                    "CUDA graph capture failed (%s); falling back to per-step "
                    "enqueue. Decoding is correct but slower; pass "
                    "--no-cuda-graphs to skip this attempt.",
                    err,
                )

        self._cross.copy_(cross_kv)
        # Zero rather than leave stale K/V: the mask makes the tail unreachable,
        # but an inf or NaN left by a previous utterance would still poison the
        # softmax through -inf + inf.
        self._cache.zero_()
        self._cache[:, :, :, :prompt_len, :].copy_(prompt_kv)
        self._pos.fill_(prompt_len)
        self._token.fill_(first_token)
        self._host_pos = prompt_len

    def can_step(self) -> bool:
        """True while the fixed-capacity cache still has room."""
        return self._host_pos < self.capacity

    @torch.no_grad()
    def step(self) -> int:
        """Advance one token and return the id it predicts.

        Replays the captured graph when there is one, and runs the identical
        body directly when capture was unavailable -- same arithmetic, same
        buffers, just without the launch saving.
        """
        if not self._ready:
            raise RuntimeError("begin() must be called before step().")
        if self._graph is not None:
            self._graph.replay()
        else:
            self._step_body()
        self._host_pos += 1
        return int(self._token.item())

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Unused. Drive the decoder via compute_cross_kv()/prefill()/step()."""
        raise NotImplementedError(
            "TextDecoderTRTKVGraph has no forward(); use prefill()/begin()/step()."
        )

    def summary(self) -> str:
        """Return a short human-readable component summary."""
        return "TensorRT KV-cached text decoder wrapper (CUDA graph replay)"


class TextDecoderEngine(nn.Module):
    """Torch module form of the Whisper decoder blocks for TRT conversion.

    Backs the single-engine "simple" decoder. Unlike the KV path, this runs
    every decoder block over the whole token prefix on each step, so it can
    be captured as one engine taking ``(x, xa, mask)``.
    """

    def __init__(self, blocks: Any) -> None:
        super().__init__()
        self.blocks = blocks

    @torch.no_grad()
    def forward(self, x: Tensor, xa: Tensor, mask: Tensor) -> Tensor:
        """Run decoder blocks for token features before output projection."""
        for block in cast(list[Any], self.blocks):
            x = block(x, xa, mask)
        return x

    def summary(self) -> str:
        """Return a short human-readable component summary."""
        return "Text decoder conversion module"


class TextDecoderTRT(nn.Module):
    """Single-engine Whisper text decoder: full recompute, no KV cache.

    The low-VRAM alternative to ``TextDecoderTRTKV`` — one TensorRT engine
    context instead of three, trading ~600 MiB of resident VRAM for slower
    decode (self-attention re-runs the whole prefix every step, O(prefix^2)).
    Only the final-position output projection is sliced, since greedy
    decoding consumes just the last token.
    """

    def __init__(self, engine: Any, state: TextDecoderState) -> None:
        super().__init__()
        self.engine = engine
        self.token_embedding = state.token_embedding
        self.positional_embedding = state.positional_embedding
        self.ln = state.ln
        self.register_buffer("mask", state.mask, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, xa: Tensor) -> Tensor:
        """Decode token ids into next-token logits (final position only)."""
        token_emb = self.token_embedding(x).to(xa.device)
        pos_emb = self.positional_embedding[: x.shape[-1]].to(xa.device)
        hidden = token_emb + pos_emb
        hidden = self.engine(hidden, xa, cast(Tensor, self.mask).to(xa.device))
        hidden = self.ln(hidden)[:, -1:, :]
        weight = self.token_embedding.weight.to(hidden.device)
        return (hidden.to(weight.dtype) @ torch.transpose(weight, 0, 1)).float()

    def summary(self) -> str:
        """Return a short human-readable component summary."""
        return "TensorRT text decoder wrapper"
