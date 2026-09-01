# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""TensorRT-backed Whisper model and model-builder utilities."""

# This module orchestrates the encoder plus the three decoder engines, their
# builders, and the loader; the decode modules themselves live in _decoder.py.
# It runs a little over the default line cap as a cohesive unit.

import ctypes
import ctypes.util
import gc
import importlib.resources
import logging
import os
import re
import time
import wave
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import tensorrt
import torch
import whisper.audio
import whisper.tokenizer
from torch import nn
from whisper import load_model
from whisper.model import LayerNorm, ModelDimensions, Tensor, disable_sdpa
from whisper.tokenizer import TO_LANGUAGE_CODE, Tokenizer

import torch2trt

from .__version__ import __version__
from ._decoder import (
    CachedDecoderStep,
    CrossKVProjector,
    DecoderEngines,
    DecodeRequest,
    PrefillDecoder,
    TextDecoderEngine,
    TextDecoderState,
    TextDecoderTRT,
    TextDecoderTRTKV,
)
from .cache import get_cache_dir, make_cache_dir

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _trt_module_class() -> type[Any]:
    """Return the torch2trt TRTModule class with runtime validation."""
    module_cls = getattr(torch2trt, "TRTModule", None)
    if module_cls is None:
        raise RuntimeError("torch2trt.TRTModule is unavailable")
    return module_cls


def _new_trt_module(device_memory: Any = None) -> Any:
    """Create a new torch2trt TRTModule instance with runtime validation.

    ``device_memory`` is an optional ``SharedDeviceMemory`` pool the module's
    execution context should draw its scratch from; ``None`` keeps torch2trt's
    default of a private pool per context.

    The keyword is omitted entirely when there is no pool: a torch2trt predating
    ``SharedDeviceMemory`` has no such parameter, and passing it would raise
    TypeError on exactly the path that is supposed to degrade quietly.
    """
    module_cls = _trt_module_class()
    if device_memory is None:
        return _instantiate_type(module_cls)
    return _instantiate_type(module_cls, device_memory=device_memory)


def _instantiate_type(class_type: type[Any], **kwargs: Any) -> Any:
    """Instantiate a type object while preserving explicit typing."""
    return class_type(**kwargs)


def _new_shared_device_memory() -> Any:
    """Return a fresh shared engine scratch pool, or None if unsupported.

    Each TensorRT execution context otherwise reserves a private device-memory
    pool sized for its engine's worst-case layer scratch. A Whisper checkpoint
    is up to four engines (encoder plus a three-engine KV decoder) that only
    ever run one at a time, so private pools reserve that scratch four times
    over. Sharing one pool sized to the largest engine gives the same execution
    with a fraction of the resident scratch.

    Returns None when running against a torch2trt without ``SharedDeviceMemory``
    so the engines simply keep their private pools rather than failing to load.
    """
    pool_cls = getattr(torch2trt, "SharedDeviceMemory", None)
    if pool_cls is None:
        logger.debug(
            "torch2trt has no SharedDeviceMemory; engines will each reserve a "
            "private device-memory pool."
        )
        return None
    # Instantiated through the typed helper, as with TRTModule above: a class
    # resolved by getattr is untyped, and calling it directly is what pylint
    # flags as not-callable.
    return _instantiate_type(pool_cls)


class IncompatibleEngineError(RuntimeError):
    """Raised when a serialized TRT engine cannot be deserialized here.

    The usual cause is a checkpoint built for a different GPU architecture
    (TensorRT plans are not portable across compute capabilities), but a
    truncated or TRT-version-mismatched plan fails the same way.
    """


def get_device_arch_tag() -> str:
    """Return a cache tag for the CUDA device this process will actually use.

    TensorRT plans are tied to the compute capability they were built on, so
    the tag becomes part of the engine filename: a plan built on an sm_86 GPU
    can never be picked up on an sm_89 one, even when both share a cache
    directory. Derived from the device torch selects rather than from
    ``nvidia-smi``, whose ordering need not match CUDA's on mixed multi-GPU
    hosts.
    """
    try:
        if not torch.cuda.is_available():
            return "smunknown"
        major, minor = torch.cuda.get_device_capability()
    except (RuntimeError, AssertionError) as err:  # pragma: no cover - driver dependent
        logger.debug("Could not query CUDA compute capability: %s", err)
        return "smunknown"
    return f"sm{major}{minor}"


def get_trt_version_tag() -> str:
    """Return a cache tag for the exact TensorRT version in use.

    Engines are built without ``VERSION_COMPATIBLE``, so a plan is loadable only
    by the TensorRT build that produced it — not merely the same major version.
    The full version therefore goes in the filename, so any upgrade (including a
    patch) quietly builds a new engine instead of failing to deserialize the old
    one. Non-alphanumerics become underscores to keep the filename plain.
    """
    return "trt" + re.sub(r"[^0-9A-Za-z]+", "_", str(tensorrt.__version__))


def _load_engine_module(
    checkpoint: dict[str, Any], key: str, what: str, device_memory: Any = None
) -> Any:
    """Deserialize one TRT engine out of a checkpoint into a TRTModule.

    The engine state is *popped* from ``checkpoint`` and dropped as soon as
    ``deserialize_cuda_engine`` has consumed it. Each entry is a host bytearray
    holding a whole serialized plan (GB-scale for the large family), and
    ``deserialize_cuda_engine`` makes its own copy — so holding every blob until
    ``load`` returns doubles the peak for no reason. Popping keeps at most one
    serialized plan live at a time.

    ``TRTModule._load_from_state_dict`` leaves ``engine`` as ``None`` when
    ``deserialize_cuda_engine`` fails, which otherwise only surfaces later as
    an opaque ``AttributeError`` on ``NoneType``. Convert it here into an
    actionable error the loader can respond to by rebuilding.
    """
    engine_state = checkpoint.pop(key)
    module = _new_trt_module(device_memory=device_memory).cuda()
    module.load_state_dict(engine_state)
    del engine_state
    _reclaim_memory()
    if getattr(module, "engine", None) is None:
        raise IncompatibleEngineError(
            f"Failed to deserialize the '{what}' TensorRT engine on this device "
            f"({get_device_arch_tag()}); the cached plan is incompatible or corrupt "
            "and must be rebuilt. See the TensorRT log above for the exact cause."
        )
    return module


def _invoke_converter(
    converter: Callable[..., Any],
    module: nn.Module,
    inputs: list[torch.Tensor],
    **kwargs: Any,
) -> Any:
    """Invoke torch2trt converter through a typed callable helper."""
    return converter(module, inputs, **kwargs)


# TensorRT reports a build that ran out of memory by naming the node whose
# tactics it could not fit, which reads as an unsupported-op or bad-export
# problem and sends people looking in the wrong place entirely. These are the
# markers seen on memory-constrained devices; matched case-insensitively.
_OOM_BUILD_MARKERS = (
    "could not find any implementation for node",
    "no implementation obeys reformatting-free rules",
    "out of memory",
    "outofmemory",
)


def _looks_like_build_oom(message: str) -> bool:
    """Return True when a TensorRT build failure reads as memory exhaustion."""
    lowered = message.lower()
    return any(marker in lowered for marker in _OOM_BUILD_MARKERS)


def _torch2trt_convert(
    module: nn.Module, inputs: list[torch.Tensor], **kwargs: Any
) -> Any:
    """Convert a Torch module to TensorRT with runtime validation."""
    converter = getattr(torch2trt, "torch2trt", None)
    if converter is None or not callable(converter):
        raise RuntimeError("torch2trt.torch2trt is unavailable")
    workspace_mb = int(kwargs.get("max_workspace_size", 0)) >> 20
    logger.debug(
        "Building TensorRT engine with a %d MiB workspace; %s.",
        workspace_mb,
        _describe_free_vram(),
    )
    try:
        return _invoke_converter(converter, module, inputs, **kwargs)
    except Exception as err:  # re-raised below; TRT's type varies by version
        if not _looks_like_build_oom(str(err)):
            raise
        raise RuntimeError(
            "TensorRT ran out of memory building this engine "
            f"(workspace {workspace_mb} MiB; {_describe_free_vram()}). "
            "TensorRT reports this as a missing implementation for a node, but "
            "the graph is supported -- there was not enough memory for the "
            "tactic search. Try a smaller model, --decoder-mode simple, or a "
            "lower --max-workspace-mb."
        ) from err


def _trt_log_level(verbose: bool) -> int:
    """Return the TensorRT logger level constant."""
    logger_cls = getattr(tensorrt, "Logger", None)
    if logger_cls is None:
        raise RuntimeError("tensorrt.Logger is unavailable")
    return logger_cls.VERBOSE if verbose else logger_cls.ERROR


def _describe_free_vram() -> str:
    """Return a short human-readable free/total VRAM string for log messages."""
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except (RuntimeError, AssertionError):
        return "free VRAM unknown"
    mib = 1 << 20
    return f"{free_bytes // mib} MiB free of {total_bytes // mib} MiB"


def _resolve_malloc_trim() -> Callable[[int], int] | None:
    """Resolve glibc's ``malloc_trim`` once, or None when unavailable.

    Only present on Linux/glibc; returns None on musl, macOS, Windows, etc.
    """
    libc_name = ctypes.util.find_library("c") or "libc.so.6"
    try:
        libc = ctypes.CDLL(libc_name)
        trim = libc.malloc_trim
    except (OSError, AttributeError):
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


_MALLOC_TRIM = _resolve_malloc_trim()


def _reclaim_memory() -> None:
    """Release freed host and GPU memory back to the OS and the CUDA driver.

    Each engine build allocates and frees GB-scale buffers in three places, so
    reclamation has to hit all three or RSS climbs across engines until the host
    OOMs (even though the memory is no longer live):

    - ``gc.collect()`` drops Python objects (the TRT builder/network/parser and
      the ONNX graph) so their memory is actually freed.
    - ``torch.cuda.empty_cache()`` returns torch's cached GPU blocks to the
      driver — TensorRT allocates straight from the driver and can't see them.
    - ``malloc_trim`` returns glibc's freed-but-retained arenas to the OS;
      without it ``free()`` keeps GB-scale build buffers in the arena and RSS
      only grows. No-op off Linux/glibc.

    Only already-free memory is released, so live modules and trace inputs are
    safe. Called between build phases and before each convert.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)


# Directory of bundled mono 16 kHz speech clips used to calibrate INT8
# activation ranges for the audio encoder. Calibrating on real speech (rather
# than the random trace input) is what makes the Q/DQ scales usable. Drop
# additional ``*.wav`` files in here to widen acoustic coverage — the more
# diverse speakers / recording conditions, the better the calibration. Clips
# must be mono, 16 kHz, 16-bit PCM.
_CALIBRATION_DIR = "calibration"


def _load_calibration_clips() -> list[np.ndarray]:
    """Load every bundled INT8-calibration clip as mono float32 PCM.

    Reads all ``*.wav`` files under :data:`_CALIBRATION_DIR` in sorted order so
    the calibration set is deterministic.
    """
    root = importlib.resources.files("whisper_trt").joinpath(_CALIBRATION_DIR)
    clips: list[np.ndarray] = []
    try:
        entries = sorted(
            (e for e in root.iterdir() if e.name.lower().endswith(".wav")),
            key=lambda e: e.name,
        )
        for entry in entries:
            with (
                importlib.resources.as_file(entry) as clip_path,
                wave.open(str(clip_path), "rb") as clip,
            ):
                frames = clip.readframes(clip.getnframes())
            clips.append(np.frombuffer(frames, dtype=np.int16).astype(np.float32))
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(
            "INT8 calibration clips are missing; cannot build an int8 engine. "
            f"Expected packaged resource directory 'whisper_trt/{_CALIBRATION_DIR}'."
        ) from exc
    if not clips:
        raise RuntimeError(
            "No INT8 calibration clips found; cannot build an int8 engine. "
            f"Add one or more 16 kHz mono WAVs under 'whisper_trt/{_CALIBRATION_DIR}'."
        )
    return [c / 32768.0 for c in clips]


def _encoder_int8_calib_dataset(
    n_mels: int, n_frames: int, positional_embedding: torch.Tensor
) -> list[list[torch.Tensor]]:
    """Build a representative INT8 calibration set for the audio encoder.

    Whisper always pads audio to a fixed 30 s window before encoding, so for
    each bundled clip we derive two real-speech mel spectrograms: the utterance
    at the start of the window (the common case) and the clip looped to fill the
    window (continuous speech). Acoustic diversity comes from the set of clips in
    :data:`_CALIBRATION_DIR`. Each item mirrors the encoder's
    ``[x, positional_embedding]`` input signature.
    """
    window = whisper.audio.N_SAMPLES
    audios: list[np.ndarray] = []
    for pcm in _load_calibration_clips():
        audios.append(pcm)  # utterance at window start
        reps = int(np.ceil(window / len(pcm)))
        audios.append(np.tile(pcm, reps)[:window])  # looped to fill the window

    dataset: list[list[torch.Tensor]] = []
    for audio in audios:
        mel = whisper.audio.log_mel_spectrogram(
            torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
            n_mels,
        )
        mel = whisper.audio.pad_or_trim(mel, n_frames).unsqueeze(0).contiguous().cuda()
        dataset.append([mel, positional_embedding])
    return dataset


@dataclass
class WhisperTRTConfig:
    """Optional runtime configuration for the WhisperTRT model."""

    tokenizer: Tokenizer | None = None
    verbose: bool = False


class _AudioEncoderEngine(nn.Module):
    """Torch module form of the Whisper audio encoder used for TRT conversion."""

    def __init__(
        self, conv1: nn.Conv1d, conv2: nn.Conv1d, blocks: Any, ln_post: LayerNorm
    ) -> None:
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.blocks = blocks
        self.ln_post = ln_post
        self._gelu = nn.GELU()

    @torch.no_grad()
    def forward(self, x: Tensor, positional_embedding: Tensor) -> Tensor:
        """Run one forward pass through the audio encoder backbone."""
        x = self._gelu(self.conv1(x))
        x = self._gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = (x + positional_embedding).to(x.dtype)
        for block in cast(list[Any], self.blocks):
            x = block(x)
        return self.ln_post(x)

    def summary(self) -> str:
        """Return a short human-readable component summary."""
        return "Audio encoder conversion module"


class AudioEncoderTRT(nn.Module):
    """Whisper audio encoder that runs through a TensorRT engine."""

    def __init__(self, engine: Any, positional_embedding: torch.Tensor) -> None:
        super().__init__()
        self.engine = engine
        self.register_buffer("positional_embedding", positional_embedding)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Encode mel frames into Whisper audio features."""
        n_audio_ctx = int(x.shape[2] // 2)
        positional_embedding = cast(torch.Tensor, self.positional_embedding)
        pos_embed = positional_embedding[-n_audio_ctx:, :]
        return self.engine(x, pos_embed)

    def summary(self) -> str:
        """Return a short human-readable component summary."""
        return "TensorRT audio encoder wrapper"


class WhisperTRT(nn.Module):
    """Whisper model optimized for TensorRT inference."""

    def __init__(
        self,
        dims: ModelDimensions,
        encoder: AudioEncoderTRT,
        decoder: TextDecoderTRTKV | TextDecoderTRT,
        config: WhisperTRTConfig | None = None,
    ) -> None:
        super().__init__()
        runtime_config = config or WhisperTRTConfig()
        self.dims = dims
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = runtime_config.tokenizer
        self.verbose = runtime_config.verbose
        self.stream = torch.cuda.Stream()

    def embed_audio(self, mel: Tensor) -> Tensor:
        """Embed mel spectrogram features with the TRT encoder."""
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: Tensor) -> Tensor:
        """Return final-position logits for the simple (single-engine) decoder."""
        return cast(TextDecoderTRT, self.decoder)(tokens, audio_features)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Unused. Use transcribe() (or embed_audio()) instead."""
        raise NotImplementedError("WhisperTRT has no forward(); use transcribe().")

    def _normalize_audio_input(self, audio: str | np.ndarray) -> np.ndarray:
        """Load path-like audio input or normalize ndarray audio input."""
        if isinstance(audio, str):
            return whisper.audio.load_audio(audio)

        audio_array = np.asarray(audio)
        if not np.issubdtype(audio_array.dtype, np.floating):
            return audio_array.astype(np.float32) / 32768.0
        return audio_array

    def _get_tokenizer(self) -> Tokenizer:
        """Return the configured tokenizer or fail with a clear error."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be configured before transcription")
        return self.tokenizer

    def _configure_tokenizer(self, tokenizer: Tokenizer, language: str) -> None:
        """Apply language and task settings to tokenizer before decoding."""
        if language.lower() != "auto":
            code = language.lower()
            if code in TO_LANGUAGE_CODE:
                code = TO_LANGUAGE_CODE[code]
            tokenizer.language = code
            logger.debug("Tokenizer language set to: %s", code)
        else:
            tokenizer.language = None
            logger.debug("Tokenizer set to auto language detection.")

        if hasattr(tokenizer, "task"):
            tokenizer.task = "transcribe"
        elif hasattr(tokenizer, "set_task"):
            tokenizer.set_task("transcribe")

    def _build_prefix_tokens(self, tokenizer: Tokenizer) -> list[int]:
        """Build special tokenizer prefix tokens for language/task/timestamps."""
        prefix_tokens: list[int] = []
        if tokenizer.language is not None:
            prefix_tokens.extend(
                tokenizer.encode(f"<|{tokenizer.language}|>", allowed_special="all")
            )

        if hasattr(tokenizer, "task") and tokenizer.task == "transcribe":
            prefix_tokens.extend(
                tokenizer.encode("<|transcribe|>", allowed_special="all")
            )

        prefix_tokens.extend(
            tokenizer.encode("<|notimestamps|>", allowed_special="all")
        )
        return prefix_tokens

    def _prepare_prompt_tokens(
        self,
        tokenizer: Tokenizer,
        audio_features: Tensor,
        max_len: int,
        initial_prompt: str | None,
    ) -> tuple[torch.Tensor, int, int]:
        """Create and seed the decoder token buffer for autoregressive decode."""
        out_tokens = torch.empty(
            (1, max_len), dtype=torch.long, device=audio_features.device
        )
        out_tokens.fill_(getattr(tokenizer, "pad", 0))

        cur_len = 0
        out_tokens[0, cur_len] = tokenizer.sot
        cur_len += 1

        for token in self._build_prefix_tokens(tokenizer):
            out_tokens[0, cur_len] = token
            cur_len += 1

        if initial_prompt:
            prompt_ids = tokenizer.encode(initial_prompt)
            # Cap the prompt to at most half the remaining buffer so there is
            # always room to decode, and keep the most recent tokens (Whisper's
            # convention). Without this an over-long prompt overflows out_tokens
            # and raises an opaque index error.
            max_prompt = max(0, (max_len - cur_len) // 2)
            if len(prompt_ids) > max_prompt:
                prompt_ids = prompt_ids[-max_prompt:] if max_prompt else []
            if prompt_ids:
                out_tokens[0, cur_len : cur_len + len(prompt_ids)] = torch.tensor(
                    prompt_ids,
                    device=audio_features.device,
                )
                cur_len += len(prompt_ids)

        return out_tokens, cur_len, cur_len

    def _prime_cache(self, request: DecodeRequest) -> tuple[Tensor, Tensor, Tensor]:
        """Prefill the KV cache from the prompt and precompute cross K/V.

        Returns the logits that predict the first generated token, the primed
        self-attention cache, and the (static) cross-attention cache.
        """
        decoder = cast(TextDecoderTRTKV, self.decoder)
        cross_kv = decoder.compute_cross_kv(request.audio_features)
        prompt_ids = request.out_tokens[0, : request.prompt_len].tolist()
        logits, self_kv = decoder.prefill(prompt_ids, cross_kv)
        return logits, self_kv, cross_kv

    def _decode_sequence(
        self,
        tokenizer: Tokenizer,
        request: DecodeRequest,
    ) -> tuple[str, list[str], float]:
        """Dispatch decoding to the cached or single-engine implementation."""
        if isinstance(self.decoder, TextDecoderTRTKV):
            return self._decode_sequence_kv(tokenizer, request)
        return self._decode_sequence_simple(tokenizer, request)

    def _decode_sequence_simple(
        self,
        tokenizer: Tokenizer,
        request: DecodeRequest,
    ) -> tuple[str, list[str], float]:
        """Autoregressively decode by recomputing the whole prefix each step.

        The single-engine decoder has no KV cache, so every step re-runs the
        full token prefix through one engine. Slower (O(prefix^2)) but uses
        one engine context instead of three.
        """
        chunks: list[str] = []
        decode_start = time.perf_counter()

        last_token_id = -1
        first_step = True
        for _ in range(request.cur_len, request.max_len):
            token_logits = self.logits(
                request.out_tokens[:, : request.cur_len],
                request.audio_features,
            )
            if first_step:
                first_step = False
                if self._is_no_speech(
                    tokenizer, token_logits, request.no_speech_threshold
                ):
                    return "", [], time.perf_counter() - decode_start
            # One GPU->CPU sync per token; reuse the int for the buffer write
            # and the stop check rather than reading the tensor back twice.
            last_token_id = int(token_logits.argmax(dim=-1)[0, -1].item())
            request.out_tokens[0, request.cur_len] = last_token_id
            request.cur_len += 1

            if request.stream:
                interim = request.out_tokens[:, request.prompt_len : request.cur_len]
                chunks.append(self._decode_tokens(interim))

            if last_token_id == tokenizer.eot:
                break

        end_index = request.cur_len
        if last_token_id == tokenizer.eot:
            end_index = request.cur_len - 1

        final_text = self._decode_tokens(
            request.out_tokens[:, request.prompt_len : end_index]
        )
        return final_text, chunks, time.perf_counter() - decode_start

    def _decode_sequence_kv(
        self,
        tokenizer: Tokenizer,
        request: DecodeRequest,
    ) -> tuple[str, list[str], float]:
        """Autoregressively decode with a self-attention KV cache.

        The prompt is replayed one token at a time to prime the cache, after
        which each generated token is fed back as a single-token step. The
        decoder reuses the precomputed cross-attention K/V throughout, so no
        prefix or encoder projection is recomputed per token.
        """
        chunks: list[str] = []
        decode_start = time.perf_counter()
        decoder = cast(TextDecoderTRTKV, self.decoder)
        logits, self_kv, cross_kv = self._prime_cache(request)

        # ``logits`` here predict the first generated token, so this is the
        # position Whisper evaluates for no-speech (see _is_no_speech).
        if self._is_no_speech(tokenizer, logits, request.no_speech_threshold):
            return "", [], time.perf_counter() - decode_start

        last_token_id = -1
        while request.cur_len < request.max_len:
            last_token_id = int(logits.argmax(dim=-1)[0, -1].item())
            request.out_tokens[0, request.cur_len] = last_token_id
            request.cur_len += 1

            if request.stream:
                interim = request.out_tokens[:, request.prompt_len : request.cur_len]
                chunks.append(self._decode_tokens(interim))

            if last_token_id == tokenizer.eot or request.cur_len >= request.max_len:
                break

            # Feed the just-generated token (at index cur_len - 1) to predict
            # the next one.
            logits, self_kv = decoder.step(
                last_token_id, request.cur_len - 1, self_kv, cross_kv
            )

        end_index = request.cur_len
        if last_token_id == tokenizer.eot:
            end_index = request.cur_len - 1

        final_text = self._decode_tokens(
            request.out_tokens[:, request.prompt_len : end_index]
        )
        return final_text, chunks, time.perf_counter() - decode_start

    def _audio_to_mel(self, audio_array: np.ndarray) -> tuple[Tensor, float]:
        """Convert normalized audio samples to a mel spectrogram tensor."""
        load_start = time.perf_counter()
        # Compute the mel on the GPU directly: avoids the CPU STFT plus a
        # full-spectrogram host->device copy (only the raw audio crosses).
        # Pass n_mels explicitly: large-v3 / large-v3-turbo use 128-mel
        # features (dims.n_mels) and the encoder engine is built for that;
        # omitting it defaults to 80 and feeds those engines the wrong shape.
        mel = whisper.audio.log_mel_spectrogram(
            audio_array,
            self.dims.n_mels,
            padding=whisper.audio.N_SAMPLES,
            device="cuda",
        )[None, ...]
        if mel.shape[2] > whisper.audio.N_FRAMES:
            mel = mel[:, :, : whisper.audio.N_FRAMES]
        return mel, time.perf_counter() - load_start

    def _decode_mel(
        self,
        mel: Tensor,
        language: str,
        stream: bool,
        initial_prompt: str | None,
        no_speech_threshold: float | None = None,
    ) -> tuple[str, list[str], float]:
        """Decode a mel tensor into final text and optional transcript chunks."""
        max_len = self.dims.n_text_ctx + 1
        # ``mel`` was produced on the caller's stream. Entering a stream context
        # does not synchronise, so without this wait the encoder can start on
        # self.stream while the mel's kernels are still in flight and read the
        # buffer a kernel short of complete. It decodes to a silence token --
        # measured at 126/2400 transcriptions of clean speech coming back as
        # '.', '[ Silence ]' or '[no audio]' across tiny/base/small, and 0/2400
        # with this wait in place.
        producer = torch.cuda.current_stream()
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(producer)
            # The mel was allocated against ``producer``; tell the allocator it
            # is live on self.stream too, or it can be handed out again while
            # the encoder is still reading it.
            mel.record_stream(self.stream)
            audio_features = self.embed_audio(mel)
            tokenizer = self._get_tokenizer()
            self._configure_tokenizer(tokenizer, language)
            out_tokens, cur_len, prompt_len = self._prepare_prompt_tokens(
                tokenizer,
                audio_features,
                max_len,
                initial_prompt,
            )
            request = DecodeRequest(
                out_tokens=out_tokens,
                cur_len=cur_len,
                prompt_len=prompt_len,
                max_len=max_len,
                audio_features=audio_features,
                stream=stream,
                no_speech_threshold=no_speech_threshold,
            )
            return self._decode_sequence(tokenizer, request)

    @torch.no_grad()
    def transcribe(
        self,
        audio: str | np.ndarray,
        language: str = "auto",
        stream: bool = False,
        initial_prompt: str | None = None,
        no_speech_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Transcribe audio with optional chunk emissions and an initial prompt.

        When ``no_speech_threshold`` is set, a window the model judges to be
        non-speech (``<|nospeech|>`` probability at or above the threshold)
        returns empty text instead of a hallucinated transcript.

        Args:
            audio: Path to an audio file or numpy array containing audio data.
            language: Language code for transcription (e.g., "en", "es") or
                "auto" for automatic language detection. Defaults to "auto".
            stream: If True, return intermediate transcription chunks in addition
                to the final text. Defaults to False.
            initial_prompt: Optional text to provide as context for the first
                transcription window, which can improve accuracy. Defaults to None.
            no_speech_threshold: Probability threshold (0.0-1.0) for suppressing
                non-speech segments. Windows with ``<|nospeech|>`` probability at
                or above this value return empty text. None disables this feature.
                Defaults to None.

        Returns:
            A dictionary containing the transcription result with a "text" key
            holding the final transcript. If stream=True, also includes a "chunks"
            key with a list of intermediate transcription segments.
        """
        start_time = time.perf_counter()
        audio_array = self._normalize_audio_input(audio)
        mel, load_time = self._audio_to_mel(audio_array)
        final_text, chunks, decode_time = self._decode_mel(
            mel,
            language,
            stream,
            initial_prompt,
            no_speech_threshold,
        )

        self.stream.synchronize()
        total_time = time.perf_counter() - start_time
        if self.verbose:
            logger.info(
                "Load & mel: %.1f ms, Decode: %.1f ms, Total: %.1f ms",
                load_time * 1000,
                decode_time * 1000,
                total_time * 1000,
            )

        # Note: intentionally no torch.cuda.empty_cache() here. Releasing the
        # allocator's cached blocks every request just forces the next request
        # to re-acquire them from the driver — it adds latency without lowering
        # steady-state VRAM, since the cached blocks are reused anyway.

        result: dict[str, Any] = {"text": final_text}
        if stream:
            result["chunks"] = chunks
        return result

    @torch.no_grad()
    def get_supported_languages(self) -> list[str]:
        """Return supported language codes, falling back to English-only."""
        tokenizer = self._get_tokenizer()
        if hasattr(tokenizer, "all_language_codes"):
            return list(tokenizer.all_language_codes)
        return ["en"]

    def _is_no_speech(
        self,
        tokenizer: Tokenizer,
        first_logits: Tensor,
        threshold: float | None,
    ) -> bool:
        """Whether the audio is silence/non-speech per the ``<|nospeech|>`` token.

        Whisper emits a dedicated ``<|nospeech|>`` token whose probability at the
        first decode position estimates how likely the window contains no speech.
        Greedy decoding never selects it (a real token always outscores it), so
        on silence the model instead hallucinates plausible-looking text
        ("www.mooji.org", "Thank you for watching", ...). Gating on this
        probability — as upstream ``whisper.transcribe`` does — suppresses those
        phantom transcripts. Returns ``False`` when the check is disabled
        (``threshold is None``) or the tokenizer lacks the token.
        """
        if threshold is None:
            return False
        no_speech_id = getattr(tokenizer, "no_speech", None)
        if no_speech_id is None:
            return False
        probs = torch.softmax(first_logits[0, -1, :].float(), dim=-1)
        return float(probs[no_speech_id].item()) >= threshold

    def _decode_tokens(self, tokens: torch.Tensor) -> str:
        """Decode token tensor to clean text without internal control markers."""
        tokenizer = self._get_tokenizer()
        text = tokenizer.decode(list(tokens.flatten().cpu().numpy()))
        return (
            text.replace("<|transcribe|>", "")
            .replace("<|notimestamps|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )


# Whisper variants whose encoder/decoder are large enough that the default
# 1 GiB tactic-search budget is too tight: they build better (and avoid the
# occasional build failure) with a more generous workspace. Used by
# ``auto_workspace_mb`` to pick a default build-time scratch budget.
LARGE_MODELS = frozenset({"large", "large-v2", "large-v3", "large-v3-turbo"})

# Per-engine workspace targets in MiB, before clamping to free VRAM.
_LARGE_WORKSPACE_MB = 4096
_DEFAULT_WORKSPACE_MB = 1024
# The workspace is reserved *concurrently* with the rest of the build, so the
# cap is taken over the VRAM left after setting aside a reserve for everything
# else the build holds at the same time: the resident Whisper weights plus
# TensorRT's constant/activation regions. Of that spare, the workspace may take
# this fraction. Subtracting the reserve first is what prevents an OOM that the
# old "fraction of total free VRAM" cap allowed — free VRAM is sampled before
# the model is loaded, so a generous workspace would otherwise leave no room for
# the weights + consts and the build would OOM mid-tactic-search anyway. The
# floor still applies — going below it starves tactic selection and raises WER.
_WORKSPACE_VRAM_FRACTION = 0.5
_BUILD_MEMORY_RESERVE_MB = 2048
_MIN_WORKSPACE_MB = 256


def auto_workspace_mb(model_name: str) -> int:
    """Choose a default TensorRT build-time workspace budget, in MiB.

    Picks a target by model size — larger models get a more generous
    tactic-search budget — then clamps it so the workspace leaves room for the
    rest of the build. The clamp sets aside ``_BUILD_MEMORY_RESERVE_MB`` of the
    currently-free VRAM for the resident weights and TensorRT's const/activation
    regions, then lets the workspace take ``_WORKSPACE_VRAM_FRACTION`` of what
    remains. This keeps a build from reserving so much scratch that the weights
    and consts can no longer fit (which OOMs the build on a small GPU). Falls
    back to the unclamped target when CUDA memory info is unavailable.

    This only chooses the *default*; an explicit ``--max-workspace-mb`` always
    takes precedence. The returned value is the build-time tactic-search
    ceiling, not a runtime allocation — see ``WhisperTRTBuilder``.

    Args:
        model_name: The Whisper model name (e.g., "tiny", "base", "small",
            "medium", "large-v2"). Determines the base workspace target, with
            larger models receiving more generous budgets.

    Returns:
        int: The build-time workspace ceiling in MiB. Returns the model-size
            target when CUDA memory info is unavailable (no device or not
            initialized); otherwise returns the minimum of the target and a
            VRAM-based cap, but never less than ``_MIN_WORKSPACE_MB``.

    Raises:
        RuntimeError: If ``torch.cuda.mem_get_info()`` is called but CUDA
            runtime encounters an error (caught and handled by falling back
            to the unclamped target).
        AssertionError: If ``torch.cuda.mem_get_info()`` is called but CUDA
            is not properly initialized (caught and handled by falling back
            to the unclamped target).
    """
    target = (
        _LARGE_WORKSPACE_MB if model_name in LARGE_MODELS else _DEFAULT_WORKSPACE_MB
    )
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
    except (RuntimeError, AssertionError):
        # No CUDA device / not initialized: trust the model-size target.
        return target
    spare_mb = free_bytes / (1 << 20) - _BUILD_MEMORY_RESERVE_MB
    cap_mb = int(_WORKSPACE_VRAM_FRACTION * spare_mb)
    return max(_MIN_WORKSPACE_MB, min(target, cap_mb))


class WhisperTRTBuilder:
    """Factory for building and loading TensorRT-backed Whisper checkpoints."""

    model: str
    fp16_mode: bool = False
    quant_mode: str = "float32"  # Options: "float32", "float16", "int8"
    decoder_mode: str = "kv"  # Options: "kv" (3 engines, fast), "simple" (1, lean)
    # Per-engine TensorRT *build-time* scratch budget. This is a ceiling on the
    # workspace a layer may use during tactic search, not a runtime allocation
    # — TensorRT reserves only what each engine actually needs (well under this
    # for these models), so it does not control resident VRAM. Keep it generous
    # so TRT can pick its best tactics; starving it (measured at 64 MiB) picked
    # worse tactics and raised WER. Lower via --max-workspace-mb only as an
    # OOM escape hatch when building a very large model.
    max_workspace_size: int = 1 << 30
    # True when max_workspace_size came from --max-workspace-mb. An explicit
    # budget is honoured as given; only the auto-selected one is re-clamped
    # per engine (see _effective_workspace).
    max_workspace_explicit: bool = False
    verbose: bool = False
    _tokenizer: Tokenizer | None = None
    _dims: ModelDimensions | None = None

    @classmethod
    def _effective_workspace(cls) -> int:
        """Return the workspace budget for the engine about to be built.

        ``auto_workspace_mb`` runs once, before the build, when nothing is
        resident yet -- no Whisper weights, no finished engines. A "kv" build
        then makes four engines in sequence, each one leaving its weights in
        VRAM, so by the last build the free memory that sized the budget is
        long gone and the stale ceiling can let tactic search reserve more than
        is left. Re-clamping against *current* free VRAM before each build is
        what keeps the later engines in a sequence from OOMing on a small GPU.

        An explicit --max-workspace-mb is returned untouched: the user asked
        for that number, and silently shrinking it would make the flag a
        suggestion.
        """
        if cls.max_workspace_explicit:
            return cls.max_workspace_size
        try:
            free_bytes, _total = torch.cuda.mem_get_info()
        except (RuntimeError, AssertionError):
            return cls.max_workspace_size
        spare_mb = free_bytes / (1 << 20) - _BUILD_MEMORY_RESERVE_MB
        cap_mb = max(_MIN_WORKSPACE_MB, int(_WORKSPACE_VRAM_FRACTION * spare_mb))
        clamped = min(cls.max_workspace_size, cap_mb << 20)
        if clamped < cls.max_workspace_size:
            logger.debug(
                "Clamping workspace from %d MiB to %d MiB for this engine (%s).",
                cls.max_workspace_size >> 20,
                clamped >> 20,
                _describe_free_vram(),
            )
        return clamped

    @classmethod
    def get_compute_type(cls) -> str:
        """Return the effective compute type based on builder configuration.

        Reconciles ``quant_mode`` and ``fp16_mode`` into a single canonical
        string used to key on-disk engine filenames. Note that "int8" describes
        the encoder only — the decoder always stays FP16 — and that the encoder
        is INT8 in its quantized layers (convolutions and projections) and FP16
        elsewhere; see ``build_audio_encoder_engine``.

        Returns:
            str: "int8" when ``quant_mode == "int8"``;
                 "float16" when ``fp16_mode`` is True;
                 "float32" otherwise.
        """
        if cls.quant_mode == "int8":
            return "int8"
        if cls.fp16_mode:
            return "float16"
        return "float32"

    @classmethod
    @torch.no_grad()
    def _load_model_once(cls) -> ModelDimensions:
        """Cache the base Whisper model's dimensions (loaded on CPU).

        Only ``.dims`` (plain metadata) is needed here, so the model is never
        moved to the GPU — a ``.cuda()`` here would reserve a full model's worth
        of VRAM in torch's cache just to read shapes, starving the build.
        """
        if cls._dims is None:
            cls._dims = load_model(cls.model).dims
        return cls._dims

    @classmethod
    def _decoder_fp16(cls) -> bool:
        """Whether decoder engines build in FP16.

        The decoder always stays FP16 (even under int8): its inputs are
        intermediate activations with no representative calibration set, and
        it is latency-bound by the autoregressive loop, not FLOP-bound.
        """
        return cls.fp16_mode or cls.quant_mode == "int8"

    @classmethod
    @torch.no_grad()
    def build_cross_kv_engine(cls) -> Any:
        """Build the engine that projects encoder features to cross K/V once."""
        dims = cls._load_model_once()
        model_inst = load_model(cls.model).cuda().eval()
        module = CrossKVProjector(model_inst.decoder.blocks)
        xa = torch.randn(1, dims.n_audio_ctx, dims.n_audio_state).cuda()
        # disable_sdpa is unnecessary here (no attention is computed) but keeps
        # the trace on whisper's plain Linear projections.
        _reclaim_memory()
        with disable_sdpa():
            return _torch2trt_convert(
                module,
                [xa],
                use_onnx=True,
                int8_mode=False,
                input_names=["xa"],
                output_names=["cross_kv"],
                max_workspace_size=cls._effective_workspace(),
                fp16_mode=cls._decoder_fp16(),
                log_level=_trt_log_level(cls.verbose),
            )

    @classmethod
    @torch.no_grad()
    def build_prefill_engine(cls) -> Any:
        """Build the prompt-prefill engine (full masked pass over the prompt).

        The prompt length (axis 1 of ``x``, both axes of ``mask``, axis 3 of
        the emitted cache) is dynamic from 1 to ``n_text_ctx``.
        """
        dims = cls._load_model_once()
        model_inst = load_model(cls.model).cuda().eval()
        module = PrefillDecoder(model_inst.decoder.blocks, dims.n_text_head)

        n_layers = dims.n_text_layer
        n_state = dims.n_text_state
        n_audio_ctx = dims.n_audio_ctx
        n_text_ctx = dims.n_text_ctx
        opt_len = max(2, min(8, n_text_ctx // 16))

        x = torch.randn(1, opt_len, n_state).cuda()
        cross_kv = torch.randn(2, n_layers, 1, n_audio_ctx, n_state).cuda()
        mask = torch.zeros(opt_len, opt_len).cuda()

        _reclaim_memory()
        with disable_sdpa():
            return _torch2trt_convert(
                module,
                [x, cross_kv, mask],
                use_onnx=True,
                int8_mode=False,
                min_shapes=[
                    (1, 1, n_state),
                    (2, n_layers, 1, n_audio_ctx, n_state),
                    (1, 1),
                ],
                opt_shapes=[
                    (1, opt_len, n_state),
                    (2, n_layers, 1, n_audio_ctx, n_state),
                    (opt_len, opt_len),
                ],
                max_shapes=[
                    (1, n_text_ctx, n_state),
                    (2, n_layers, 1, n_audio_ctx, n_state),
                    (n_text_ctx, n_text_ctx),
                ],
                input_names=["x", "cross_kv", "mask"],
                output_names=["last_hidden", "self_kv"],
                max_workspace_size=cls._effective_workspace(),
                fp16_mode=cls._decoder_fp16(),
                log_level=_trt_log_level(cls.verbose),
            )

    @classmethod
    @torch.no_grad()
    def build_decoder_step_engine(cls) -> Any:
        """Build the single-token KV-cached decoder-step engine.

        The self-attention cache length (axis 3 of ``self_kv``) is dynamic. It
        starts at the prompt length (>= 1; the prefill engine seeds it, so the
        step engine never sees an empty cache — TensorRT cannot bind a
        zero-length input) and grows to ``n_text_ctx``.
        """
        dims = cls._load_model_once()
        model_inst = load_model(cls.model).cuda().eval()
        module = CachedDecoderStep(model_inst.decoder.blocks, dims.n_text_head)

        n_layers = dims.n_text_layer
        n_state = dims.n_text_state
        n_audio_ctx = dims.n_audio_ctx
        n_text_ctx = dims.n_text_ctx
        opt_past = max(1, n_text_ctx // 16)

        x = torch.randn(1, 1, n_state).cuda()
        self_kv = torch.randn(2, n_layers, 1, opt_past, n_state).cuda()
        cross_kv = torch.randn(2, n_layers, 1, n_audio_ctx, n_state).cuda()

        _reclaim_memory()
        with disable_sdpa():
            return _torch2trt_convert(
                module,
                [x, self_kv, cross_kv],
                use_onnx=True,
                int8_mode=False,
                min_shapes=[
                    (1, 1, n_state),
                    (2, n_layers, 1, 1, n_state),
                    (2, n_layers, 1, n_audio_ctx, n_state),
                ],
                opt_shapes=[
                    (1, 1, n_state),
                    (2, n_layers, 1, opt_past, n_state),
                    (2, n_layers, 1, n_audio_ctx, n_state),
                ],
                max_shapes=[
                    (1, 1, n_state),
                    (2, n_layers, 1, n_text_ctx, n_state),
                    (2, n_layers, 1, n_audio_ctx, n_state),
                ],
                input_names=["x", "self_kv", "cross_kv"],
                output_names=["hidden", "new_self_kv"],
                max_workspace_size=cls._effective_workspace(),
                fp16_mode=cls._decoder_fp16(),
                log_level=_trt_log_level(cls.verbose),
            )

    @classmethod
    @torch.no_grad()
    def build_text_decoder_engine(cls) -> Any:
        """Build the single-engine ("simple") text decoder.

        One engine for the whole decoder forward; the prefix is recomputed
        each step (no KV cache). This is the low-VRAM alternative selected by
        ``decoder_mode == "simple"`` — one engine context instead of the
        three the cached decoder needs.
        """
        dims = cls._load_model_once()
        model_inst = load_model(cls.model).cuda().eval()
        decoder_blocks_module = TextDecoderEngine(model_inst.decoder.blocks)
        x = torch.randn(1, 1, dims.n_text_state).cuda()
        xa = torch.randn(1, dims.n_audio_ctx, dims.n_audio_state).cuda()
        mask = torch.randn(dims.n_text_ctx, dims.n_text_ctx).cuda()
        # Whisper's SDPA path computes is_causal as `mask is not None and
        # n_ctx > 1`; under the ONNX trace n_ctx is symbolic, so that
        # expression is a Tensor and SDPA rejects it (is_causal must be a
        # bool). Convert via whisper's manual-attention path instead — it is
        # mathematically identical and applies the mask explicitly.
        _reclaim_memory()
        with disable_sdpa():
            return _torch2trt_convert(
                decoder_blocks_module,
                [x, xa, mask],
                use_onnx=True,
                int8_mode=False,
                min_shapes=[
                    (1, 1, dims.n_text_state),
                    (1, 1, dims.n_audio_state),
                    (dims.n_text_ctx, dims.n_text_ctx),
                ],
                opt_shapes=[
                    (1, 1, dims.n_text_state),
                    (1, dims.n_audio_ctx, dims.n_audio_state),
                    (dims.n_text_ctx, dims.n_text_ctx),
                ],
                max_shapes=[
                    (1, dims.n_text_ctx, dims.n_text_state),
                    (1, dims.n_audio_ctx, dims.n_audio_state),
                    (dims.n_text_ctx, dims.n_text_ctx),
                ],
                input_names=["x", "xa", "mask"],
                output_names=["output"],
                max_workspace_size=cls._effective_workspace(),
                fp16_mode=cls._decoder_fp16(),
                log_level=_trt_log_level(cls.verbose),
            )

    @classmethod
    @torch.no_grad()
    def build_audio_encoder_engine(cls) -> Any:
        """Build and return a TensorRT audio encoder engine."""
        dims = cls._load_model_once()
        model_inst = load_model(cls.model).cuda().eval()
        encoder_module = _AudioEncoderEngine(
            model_inst.encoder.conv1,
            model_inst.encoder.conv2,
            model_inst.encoder.blocks,
            model_inst.encoder.ln_post,
        )
        n_frames = dims.n_audio_ctx * 2
        x = torch.randn(1, dims.n_mels, n_frames).cuda()
        positional_embedding = cast(
            torch.Tensor,
            model_inst.encoder.positional_embedding,
        )
        if not positional_embedding.is_cuda:
            positional_embedding = positional_embedding.cuda()
        positional_embedding = positional_embedding.detach()
        int8_mode = cls.quant_mode == "int8"
        # Trace through whisper's manual-attention path (not SDPA) for the
        # same reason as build_text_decoder_engine.
        _reclaim_memory()
        # Explicit quantization: torch2trt rewrites the exported ONNX graph with
        # Q/DQ pairs calibrated on these mels, so the graph — not a builder flag
        # TensorRT is free to ignore — is what makes the engine INT8. Real speech
        # rather than the random trace input is what makes the ranges usable.
        int8_calib_dataset = (
            _encoder_int8_calib_dataset(dims.n_mels, n_frames, positional_embedding)
            if int8_mode
            else None
        )
        with disable_sdpa():
            return _torch2trt_convert(
                encoder_module,
                [x, positional_embedding],
                use_onnx=True,
                int8_mode=int8_mode,
                int8_calib_dataset=int8_calib_dataset,
                min_shapes=[(1, dims.n_mels, 1), (1, dims.n_audio_state)],
                opt_shapes=[
                    (1, dims.n_mels, n_frames),
                    (dims.n_audio_ctx, dims.n_audio_state),
                ],
                max_shapes=[
                    (1, dims.n_mels, n_frames),
                    (dims.n_audio_ctx, dims.n_audio_state),
                ],
                input_names=["x", "positional_embedding"],
                output_names=["output"],
                max_workspace_size=cls._effective_workspace(),
                # Keep the residual adds and layer norms out of INT8. They
                # carry the accumulated activations, whose outliers set an
                # enormous per-tensor range -- the worst scale measured was
                # 4.35, i.e. a span of ~553 squeezed into 256 levels -- and
                # quantizing them is what destroys the encoder. Measured on
                # base, encoder features against the FP16 engine:
                #
                #   quantize everything        cosine 0.61  -> "" / "music"
                #   without Add + LayerNorm    cosine 0.98  -> correct text
                #
                # They cost nearly nothing to leave in FP16 (elementwise, not
                # the compute), so the MatMuls carrying the INT8 win stay
                # quantized. Conv adds ~0.001 and is left in.
                #
                # ModelOpt already keeps the attention BMMs out on its own
                # (only weight-bearing MatMuls get Q/DQ), so they need no entry
                # here -- and excluding them explicitly changes nothing.
                int8_op_block_list=("Add", "LayerNormalization"),
                # Everything left unquantized is cast to FP16 in the same
                # quantizer pass, so an int8 encoder is INT8-where-quantized,
                # FP16 elsewhere.
                fp16_mode=cls.fp16_mode or int8_mode,
                log_level=_trt_log_level(cls.verbose),
            )

    @classmethod
    @torch.no_grad()
    def get_text_decoder_extra_state(cls, model_inst: Any = None) -> dict[str, Any]:
        """Return non-engine text-decoder state needed at runtime.

        Pass an already-loaded (CPU) model to reuse it; otherwise one is loaded.
        Only small params/buffers are read out for the checkpoint (re-homed to
        CUDA at load time), so the model stays on CPU.
        """
        if model_inst is None:
            model_inst = load_model(cls.model)
        return {
            "token_embedding": model_inst.decoder.token_embedding.state_dict(),
            "positional_embedding": model_inst.decoder.positional_embedding,
            "ln": model_inst.decoder.ln.state_dict(),
            "mask": model_inst.decoder.mask,
        }

    @classmethod
    @torch.no_grad()
    def get_audio_encoder_extra_state(cls, model_inst: Any = None) -> dict[str, Any]:
        """Return non-engine audio-encoder state needed at runtime.

        Pass an already-loaded (CPU) model to reuse it; otherwise one is loaded.
        """
        if model_inst is None:
            model_inst = load_model(cls.model)
        return {"positional_embedding": model_inst.encoder.positional_embedding}

    @classmethod
    @torch.no_grad()
    def _build_decoder_engines(cls) -> dict[str, Any]:
        """Build the decoder engine(s) for the active ``decoder_mode``.

        "kv" yields the three cached-decode engines; "simple" yields the one
        full-recompute engine. The runtime decoder is reconstructed from
        whichever keys are present (the checkpoint records the mode).
        """
        if cls.decoder_mode not in _ENGINE_SCHEMA:
            raise RuntimeError(
                f"Unknown decoder_mode '{cls.decoder_mode}'. "
                f"Valid options: {list(_ENGINE_SCHEMA.keys())}"
            )
        if cls.decoder_mode == "simple":
            engines = {
                "text_decoder_engine": cls.build_text_decoder_engine().state_dict()
            }
            _reclaim_memory()
            return engines
        # Build sequentially, reclaiming each engine's host-side build buffers
        # (ONNX protobuf, TRT parser/builder) before the next load_model, so a
        # finished engine's memory doesn't overlap the next engine's
        # model-sized load transient and inflate the peak.
        engines = {"cross_kv_engine": cls.build_cross_kv_engine().state_dict()}
        _reclaim_memory()
        engines["prefill_engine"] = cls.build_prefill_engine().state_dict()
        _reclaim_memory()
        engines["decoder_step_engine"] = cls.build_decoder_step_engine().state_dict()
        _reclaim_memory()
        return engines

    @classmethod
    @torch.no_grad()
    def build(cls, output_path: str, verbose: bool = False) -> None:
        """Build and persist a TensorRT checkpoint for this Whisper variant."""
        cls.verbose = verbose
        # Harvest all non-engine state (dims + the small decoder/encoder
        # tensors) from a single up-front model load while host RAM is still
        # free, then release the model before the memory-heavy engine builds.
        # Previously each extra-state read did its own full-model load, and the
        # audio one ran after every engine was built and held for the
        # checkpoint — a model-sized host transient on top of the peak that
        # OOM-killed RAM-constrained hosts. ``del base`` frees the bulk weights;
        # the small harvested tensors are kept alive by the dicts below.
        base = load_model(cls.model)
        cls._dims = base.dims  # cache so the build methods don't reload for dims
        checkpoint: dict[str, Any] = {
            "whisper_trt_version": __version__,
            "dims": asdict(base.dims),
            "decoder_mode": cls.decoder_mode,
            "text_decoder_extra_state": cls.get_text_decoder_extra_state(base),
            "audio_encoder_extra_state": cls.get_audio_encoder_extra_state(base),
        }
        del base
        _reclaim_memory()
        checkpoint.update(cls._build_decoder_engines())
        checkpoint["audio_encoder_engine"] = (
            cls.build_audio_encoder_engine().state_dict()
        )
        _reclaim_memory()
        torch.save(checkpoint, output_path)

    @classmethod
    def get_tokenizer(cls) -> Tokenizer:
        """Return tokenizer associated with this model family."""
        if cls._tokenizer is None:
            model_inst = load_model(cls.model)
            cls._tokenizer = whisper.tokenizer.get_tokenizer(
                model_inst.is_multilingual,
                num_languages=model_inst.num_languages,
                language=None,
                task="transcribe",
            )
        return cls._tokenizer

    @classmethod
    def _load_audio_encoder(
        cls,
        checkpoint: dict[str, Any],
        device_memory: Any = None,
    ) -> AudioEncoderTRT:
        """Construct AudioEncoderTRT from a serialized checkpoint."""
        audio_encoder_engine = _load_engine_module(
            checkpoint, "audio_encoder_engine", "audio_encoder", device_memory
        )
        audio_state = checkpoint["audio_encoder_extra_state"]
        return AudioEncoderTRT(
            audio_encoder_engine,
            audio_state["positional_embedding"],
        )

    @classmethod
    def _load_text_decoder_state(
        cls,
        checkpoint: dict[str, Any],
        dims: ModelDimensions,
    ) -> TextDecoderState:
        """Reconstruct the shared (non-engine) decoder state from a checkpoint."""
        text_state = checkpoint["text_decoder_extra_state"]
        token_embedding = nn.Embedding(dims.n_vocab, dims.n_text_state)
        token_embedding.load_state_dict(text_state["token_embedding"])
        if cls._decoder_fp16():
            # This table is the largest thing torch itself keeps resident: the
            # multilingual vocab makes it n_vocab x n_text_state, which is
            # ~265 MiB in FP32 for the large family. It is tied weights — used
            # both for the input lookup and for the output logits projection —
            # and the decoder engines it feeds are FP16 either way, so keeping a
            # FP32 master copy buys nothing but VRAM. The consumers in
            # ``_decoder`` cast the hidden state to the weight dtype, so this is
            # dtype-safe in both modes; logits are still returned as FP32.
            token_embedding = token_embedding.half()
        positional_embedding = nn.Parameter(text_state["positional_embedding"]).cuda()
        ln_layer = LayerNorm(dims.n_text_state)
        ln_layer.load_state_dict(text_state["ln"])
        mask = text_state["mask"]
        if not mask.is_cuda:
            mask = mask.cuda()
        return TextDecoderState(
            token_embedding=token_embedding,
            positional_embedding=positional_embedding,
            ln=ln_layer,
            mask=mask,
        )

    @classmethod
    def _load_text_decoder(
        cls,
        checkpoint: dict[str, Any],
        dims: ModelDimensions,
        device_memory: Any = None,
    ) -> TextDecoderTRTKV | TextDecoderTRT:
        """Construct the runtime text decoder matching the checkpoint's mode."""
        state = cls._load_text_decoder_state(checkpoint, dims)
        if checkpoint.get("decoder_mode") == "simple":
            engine = _load_engine_module(
                checkpoint, "text_decoder_engine", "text_decoder", device_memory
            )
            return TextDecoderTRT(engine, state)

        cross_kv_engine = _load_engine_module(
            checkpoint, "cross_kv_engine", "cross_kv", device_memory
        )
        prefill_engine = _load_engine_module(
            checkpoint, "prefill_engine", "prefill", device_memory
        )
        step_engine = _load_engine_module(
            checkpoint, "decoder_step_engine", "decoder_step", device_memory
        )
        return TextDecoderTRTKV(
            DecoderEngines(
                cross_kv=cross_kv_engine,
                prefill=prefill_engine,
                step=step_engine,
            ),
            state,
            dims,
        )

    @classmethod
    @torch.no_grad()
    def load(cls, trt_model_path: str) -> WhisperTRT:
        """Load a TensorRT checkpoint from disk into a ready-to-run model."""
        # map_location="cpu": every tensor in the checkpoint is either small
        # (the harvested embeddings/masks, moved to the device explicitly below)
        # or a host-side serialized plan. Letting torch restore them onto the
        # GPU only to have the engines' own device allocations land on top of
        # them raises the load-time peak for nothing.
        # weights_only=True: a checkpoint is an untrusted file on disk, and
        # unrestricted unpickling would execute whatever it carries. Everything
        # these hold survives the restricted unpickler -- tensors, dicts,
        # OrderedDicts, primitives, and the bytearray TensorRT plans.
        checkpoint = torch.load(trt_model_path, map_location="cpu", weights_only=True)
        dims = ModelDimensions(**checkpoint["dims"])
        # One scratch pool for every engine in this checkpoint. The pool is
        # referenced by each TRTModule, so it stays alive as long as the
        # contexts that point into it — it must not outlive them or be dropped
        # first. Engines run strictly one at a time here (encode, then the
        # decode loop), which is the condition sharing requires.
        device_memory = _new_shared_device_memory()
        encoder = cls._load_audio_encoder(checkpoint, device_memory)
        decoder = cls._load_text_decoder(checkpoint, dims, device_memory)
        # The engines are deserialized; only the small extra-state dicts remain.
        del checkpoint
        _reclaim_memory()
        if device_memory is not None:
            logger.debug(
                "Engine scratch: %.1f MiB shared across all engines.",
                device_memory.nbytes / (1 << 20),
            )

        whisper_trt = WhisperTRT(
            dims,
            encoder,
            decoder,
            WhisperTRTConfig(tokenizer=cls.get_tokenizer(), verbose=cls.verbose),
        )
        return whisper_trt.cuda().eval()


class EnBuilder(WhisperTRTBuilder):
    """Builder for English-only Whisper model variants."""

    @classmethod
    def get_tokenizer(cls) -> Tokenizer:
        """Return the tokenizer configured for English transcription."""
        return whisper.tokenizer.get_tokenizer(
            multilingual=False,
            num_languages=99,
            language="en",
            task="transcribe",
        )


class TinyEnBuilder(EnBuilder):
    """Builder for the tiny English-only Whisper TRT model."""

    model: str = "tiny.en"


class BaseEnBuilder(EnBuilder):
    """Builder for the base English-only Whisper TRT model."""

    model: str = "base.en"


class SmallEnBuilder(EnBuilder):
    """Builder for the small English-only Whisper TRT model."""

    model: str = "small.en"


class TinyBuilder(WhisperTRTBuilder):
    """Builder for the tiny multilingual Whisper TRT model."""

    model: str = "tiny"


class BaseBuilder(WhisperTRTBuilder):
    """Builder for the base multilingual Whisper TRT model."""

    model: str = "base"


class SmallBuilder(WhisperTRTBuilder):
    """Builder for the small multilingual Whisper TRT model."""

    model: str = "small"


class MediumBuilder(WhisperTRTBuilder):
    """Builder for the medium multilingual Whisper TRT model."""

    model: str = "medium"


class LargeBuilder(WhisperTRTBuilder):
    """Builder for the large multilingual Whisper TRT model."""

    model: str = "large"


class LargeV2Builder(WhisperTRTBuilder):
    """Builder for the large-v2 multilingual Whisper TRT model."""

    model: str = "large-v2"


class LargeV3Builder(WhisperTRTBuilder):
    """Builder for the large-v3 multilingual Whisper TRT model."""

    model: str = "large-v3"


class LargeV3TurboBuilder(WhisperTRTBuilder):
    """Builder for the large-v3-turbo multilingual Whisper TRT model."""

    model: str = "large-v3-turbo"


MODEL_FILENAMES = {
    "tiny.en": "tiny_en_trt.pth",
    "base.en": "base_en_trt.pth",
    "small.en": "small_en_trt.pth",
    "tiny": "tiny_trt.pth",
    "base": "base_trt.pth",
    "small": "small_trt.pth",
    "medium": "medium_trt.pth",
    "large": "large_trt.pth",
    "large-v2": "large_v2_trt.pth",
    "large-v3": "large_v3_trt.pth",
    "large-v3-turbo": "large_v3_turbo_trt.pth",
}

MODEL_BUILDERS = {
    "tiny.en": TinyEnBuilder,
    "base.en": BaseEnBuilder,
    "small.en": SmallEnBuilder,
    "tiny": TinyBuilder,
    "base": BaseBuilder,
    "small": SmallBuilder,
    "medium": MediumBuilder,
    "large": LargeBuilder,
    "large-v2": LargeV2Builder,
    "large-v3": LargeV3Builder,
    "large-v3-turbo": LargeV3TurboBuilder,
}


# Per-decoder-mode on-disk engine layout tag. Bump a value when that mode's
# serialized layout changes so stale checkpoints are rebuilt rather than
# mis-loaded; the two modes also get distinct tags so their caches never
# collide. "kv4" = the three-engine KV-cached decoder (cross_kv + prefill +
# step); "simple1" = the single full-recompute decoder engine. Pre-schema
# engines used no tag, so those files remain on disk untouched.
_ENGINE_SCHEMA = {"kv": "kv4", "simple": "simple1"}


def get_model_filename(
    name: str,
    quant_mode: str,
    decoder_mode: str | None = None,
    max_workspace_mb: int | None = None,
) -> str:
    """
    Returns the compute-type- and decoder-mode-aware cached engine filename.

    Each distinct compute type (float32, float16, int8), decoder mode (kv,
    simple), workspace budget, and GPU architecture produces a separate cached
    engine file, preventing silent reuse of an engine built under different
    settings. The engine-schema tag additionally invalidates caches whose
    serialized layout no longer matches the loader, the ``sm<cc>`` tag keeps
    a plan built on one compute capability from being loaded on another (TRT
    deserialize "incompatible device" error 6) even when several machines share
    one cache directory, and the ``trt<version>`` tag does the same across
    TensorRT builds, whose plans are only loadable by the exact version that
    produced them.

    Args:
        name (str): The model name (e.g. "tiny", "base.en").
        quant_mode (str): The quantization mode ("float32", "float16", or "int8").
        decoder_mode (str | None): "kv" or "simple"; defaults to the builder's
            current ``decoder_mode``.
        max_workspace_mb (int | None): Per-engine TensorRT workspace budget in MiB;
            defaults to the builder's current ``max_workspace_size`` converted to MiB.

    Returns:
        str: Filename with the quant mode, schema, workspace, GPU architecture,
            and TensorRT version embedded
            (e.g. "tiny_trt_float16_kv4_ws1024_sm89_trt11_2_1_2.pth").

    Raises:
        RuntimeError: If ``name`` is not a recognised model name or if
            ``decoder_mode`` is not a valid decoder mode.
    """
    if name not in MODEL_FILENAMES:
        raise RuntimeError(f"Model '{name}' is not supported by WhisperTRT.")
    mode = decoder_mode or WhisperTRTBuilder.decoder_mode
    if mode not in _ENGINE_SCHEMA:
        raise RuntimeError(
            f"Unknown decoder_mode '{mode}'. Valid options: {list(_ENGINE_SCHEMA.keys())}"
        )
    schema = _ENGINE_SCHEMA[mode]
    workspace_mb = (
        max_workspace_mb
        if max_workspace_mb is not None
        else (WhisperTRTBuilder.max_workspace_size >> 20)
    )
    base = MODEL_FILENAMES[name]
    stem, ext = os.path.splitext(base)
    return (
        f"{stem}_{quant_mode}_{schema}_ws{workspace_mb}"
        f"_{get_device_arch_tag()}_{get_trt_version_tag()}{ext}"
    )


def load_trt_model(
    name: str,
    path: str | None = None,
    build: bool = True,
    verbose: bool = False,
    language: str = "auto",
) -> WhisperTRT:
    """Load (or build and then load) a TensorRT Whisper model by name."""
    logger.debug(
        "Loading TRT model '%s' with compute_type=%s (quant_mode=%s, fp16_mode=%s)",
        name,
        WhisperTRTBuilder.get_compute_type(),
        WhisperTRTBuilder.quant_mode,
        WhisperTRTBuilder.fp16_mode,
    )

    if name not in MODEL_BUILDERS:
        raise RuntimeError(f"Model '{name}' is not supported by WhisperTRT.")
    # determine on-disk path — include quant_mode and workspace in filename to avoid
    # silent reuse of an engine built under different settings.

    if path is None:
        filename = get_model_filename(
            name,
            WhisperTRTBuilder.get_compute_type(),
            max_workspace_mb=WhisperTRTBuilder.max_workspace_size >> 20,
        )
        path = os.path.join(get_cache_dir(), filename)
        make_cache_dir()

    builder = MODEL_BUILDERS[name]
    if not os.path.exists(path):
        if not build:
            raise RuntimeError(f"No model found at {path}; pass build=True.")
        builder.build(path, verbose=verbose)
        # A build in this process leaves GB-scale residue behind: the ONNX
        # graphs and TRT builder arenas on the host, and torch's cached device
        # blocks from tracing the FP16 model. None of it is live, but without an
        # explicit reclaim the engine we are about to load comes up on top of
        # it, and the process holds a build-shaped footprint for its whole
        # lifetime rather than an inference-shaped one.
        _reclaim_memory()

    try:
        trt_model = builder.load(path)
    except IncompatibleEngineError as err:
        # The cached plan cannot run here (built for another GPU arch, or a
        # different TRT version, or truncated). Filenames are arch-keyed, so
        # this should be rare, but discard the unusable file and rebuild once
        # rather than dying on a cache we know is dead.
        if not build:
            raise
        logger.warning("Rebuilding unusable TRT checkpoint at %s: %s", path, err)
        # Engine checkpoints are multi-GB, so drop the dead copy rather than
        # renaming it aside. missing_ok: another process sharing this cache dir
        # may have reached the same conclusion and unlinked it first.
        Path(path).unlink(missing_ok=True)
        builder.build(path, verbose=verbose)
        _reclaim_memory()
        trt_model = builder.load(path)

    try:
        silence = np.zeros((whisper.audio.N_SAMPLES,), dtype=np.float32)
        _ = trt_model.transcribe(silence, language=language, stream=False)
    except (RuntimeError, ValueError) as err:
        logger.debug("Warm-up skipped: %s", err)

    return trt_model
