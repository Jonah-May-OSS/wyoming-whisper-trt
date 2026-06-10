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
# pylint: disable=too-many-lines

import importlib.resources
import logging
import os
import time
import wave
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import tensorrt
import torch
import whisper.audio
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
    TextDecoderState,
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


def _new_trt_module() -> Any:
    """Create a new torch2trt TRTModule instance with runtime validation."""
    return _instantiate_type(_trt_module_class())


def _instantiate_type(class_type: type[Any]) -> Any:
    """Instantiate a type object while preserving explicit typing."""
    return class_type()


def _invoke_converter(
    converter: Callable[..., Any],
    module: nn.Module,
    inputs: list[torch.Tensor],
    **kwargs: Any,
) -> Any:
    """Invoke torch2trt converter through a typed callable helper."""
    return converter(module, inputs, **kwargs)


def _torch2trt_convert(
    module: nn.Module, inputs: list[torch.Tensor], **kwargs: Any
) -> Any:
    """Convert a Torch module to TensorRT with runtime validation."""
    converter = getattr(torch2trt, "torch2trt", None)
    if converter is None or not callable(converter):
        raise RuntimeError("torch2trt.torch2trt is unavailable")
    return _invoke_converter(converter, module, inputs, **kwargs)


def _trt_log_level(verbose: bool) -> int:
    """Return the TensorRT logger level constant."""
    logger_cls = getattr(tensorrt, "Logger", None)
    if logger_cls is None:
        raise RuntimeError("tensorrt.Logger is unavailable")
    return logger_cls.VERBOSE if verbose else logger_cls.ERROR


# Directory of bundled mono 16 kHz speech clips used to calibrate INT8
# activation ranges for the audio encoder. Calibrating on real speech (rather
# than the random trace inputs torch2trt falls back to) is what makes INT8
# produce usable scales. Drop additional ``*.wav`` files in here to widen
# acoustic coverage — the more diverse speakers / recording conditions, the
# better the calibration. Clips must be mono, 16 kHz, 16-bit PCM.
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
        decoder: TextDecoderTRTKV,
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
        decoder = self.decoder
        cross_kv = decoder.compute_cross_kv(request.audio_features)
        prompt_ids = request.out_tokens[0, : request.prompt_len].tolist()
        logits, self_kv = decoder.prefill(prompt_ids, cross_kv)
        return logits, self_kv, cross_kv

    def _decode_sequence(
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
        logits, self_kv, cross_kv = self._prime_cache(request)

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
            logits, self_kv = self.decoder.step(
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
        mel = whisper.audio.log_mel_spectrogram(
            audio_array,
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
    ) -> tuple[str, list[str], float]:
        """Decode a mel tensor into final text and optional transcript chunks."""
        max_len = self.dims.n_text_ctx + 1
        with torch.cuda.stream(self.stream):
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
            )
            return self._decode_sequence(tokenizer, request)

    @torch.no_grad()
    def transcribe(
        self,
        audio: str | np.ndarray,
        language: str = "auto",
        stream: bool = False,
        initial_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe audio with optional chunk emissions and an initial prompt."""
        start_time = time.perf_counter()
        audio_array = self._normalize_audio_input(audio)
        mel, load_time = self._audio_to_mel(audio_array)
        final_text, chunks, decode_time = self._decode_mel(
            mel,
            language,
            stream,
            initial_prompt,
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


class WhisperTRTBuilder:
    """Factory for building and loading TensorRT-backed Whisper checkpoints."""

    model: str
    fp16_mode: bool = False
    quant_mode: str = "float32"  # Options: "float32", "float16", "int8"
    # Per-engine TensorRT scratch ceiling. The decoder is three engines now,
    # each reserving workspace from this budget, so an over-generous limit
    # multiplies runtime VRAM. 256 MiB is ample for these small engines
    # (torch2trt itself defaults to 32 MiB); raise via --max-workspace-mb if a
    # large model needs more. Changing it invalidates cached engines.
    max_workspace_size: int = 1 << 28
    verbose: bool = False
    _tokenizer: Tokenizer | None = None
    _dims: ModelDimensions | None = None

    @classmethod
    def get_compute_type(cls) -> str:
        """Return the effective compute type based on builder configuration.

        Reconciles ``quant_mode`` and ``fp16_mode`` into a single canonical
        string used to key on-disk engine filenames. Note that "int8" denotes
        a *request* for an INT8-calibrated encoder (the decoder always stays
        FP16); TensorRT implicit quantization is per-layer optional, so the
        built engine may contain no INT8 layers at all — see
        ``build_audio_encoder_engine``.

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
        """Load base Whisper model once per builder class and cache dimensions."""
        if cls._dims is None:
            cls._dims = load_model(cls.model).cuda().eval().dims
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
        with disable_sdpa():
            return _torch2trt_convert(
                module,
                [xa],
                use_onnx=True,
                int8_mode=False,
                input_names=["xa"],
                output_names=["cross_kv"],
                max_workspace_size=cls.max_workspace_size,
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
                max_workspace_size=cls.max_workspace_size,
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
                max_workspace_size=cls.max_workspace_size,
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
        # Calibrate INT8 activation ranges on real speech mels rather than
        # letting torch2trt fall back to the random trace inputs.
        #
        # Caveat (measured): this is TensorRT *implicit* quantization, which
        # treats INT8 as optional per layer — TensorRT only picks an INT8
        # tactic where it times faster than FP16. On TensorRT 10.16 / RTX
        # 3050 with model 'base', every layer came out FP16 and the engine
        # was identical to a float16 build (verify with script/layer_report).
        # Implicit quantization is deprecated in TensorRT 10; guaranteed INT8
        # would need explicit Q/DQ quantization, worthwhile only for
        # medium/large encoders.
        int8_calib_dataset = (
            _encoder_int8_calib_dataset(dims.n_mels, n_frames, positional_embedding)
            if int8_mode
            else None
        )
        # Trace through whisper's manual-attention path (not SDPA) for the
        # same reason as build_text_decoder_engine.
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
                max_workspace_size=cls.max_workspace_size,
                # Allow FP16 fallback alongside INT8 so TensorRT can keep
                # precision-sensitive layers in FP16 (mixed precision) instead
                # of forcing every layer to INT8.
                fp16_mode=cls.fp16_mode or int8_mode,
                log_level=_trt_log_level(cls.verbose),
            )

    @classmethod
    @torch.no_grad()
    def get_text_decoder_extra_state(cls) -> dict[str, Any]:
        """Return non-engine text-decoder state needed at runtime."""
        model_inst = load_model(cls.model).cuda().eval()
        return {
            "token_embedding": model_inst.decoder.token_embedding.state_dict(),
            "positional_embedding": model_inst.decoder.positional_embedding,
            "ln": model_inst.decoder.ln.state_dict(),
            "mask": model_inst.decoder.mask,
        }

    @classmethod
    @torch.no_grad()
    def get_audio_encoder_extra_state(cls) -> dict[str, Any]:
        """Return non-engine audio-encoder state needed at runtime."""
        model_inst = load_model(cls.model).cuda().eval()
        return {"positional_embedding": model_inst.encoder.positional_embedding}

    @classmethod
    @torch.no_grad()
    def build(cls, output_path: str, verbose: bool = False) -> None:
        """Build and persist a TensorRT checkpoint for this Whisper variant."""
        cls.verbose = verbose
        checkpoint = {
            "whisper_trt_version": __version__,
            "dims": asdict(load_model(cls.model).dims),
            "cross_kv_engine": cls.build_cross_kv_engine().state_dict(),
            "prefill_engine": cls.build_prefill_engine().state_dict(),
            "decoder_step_engine": cls.build_decoder_step_engine().state_dict(),
            "text_decoder_extra_state": cls.get_text_decoder_extra_state(),
            "audio_encoder_engine": cls.build_audio_encoder_engine().state_dict(),
            "audio_encoder_extra_state": cls.get_audio_encoder_extra_state(),
        }
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
        return cast(Tokenizer, cls._tokenizer)

    @classmethod
    def _load_audio_encoder(
        cls,
        checkpoint: dict[str, Any],
    ) -> AudioEncoderTRT:
        """Construct AudioEncoderTRT from a serialized checkpoint."""
        audio_encoder_engine = _new_trt_module().cuda()
        audio_encoder_engine.load_state_dict(checkpoint["audio_encoder_engine"])
        audio_state = checkpoint["audio_encoder_extra_state"]
        return AudioEncoderTRT(
            audio_encoder_engine,
            audio_state["positional_embedding"],
        )

    @classmethod
    def _load_text_decoder(
        cls,
        checkpoint: dict[str, Any],
        dims: ModelDimensions,
    ) -> TextDecoderTRTKV:
        """Construct the KV-cached text decoder from a serialized checkpoint."""
        cross_kv_engine = _new_trt_module().cuda()
        cross_kv_engine.load_state_dict(checkpoint["cross_kv_engine"])
        prefill_engine = _new_trt_module().cuda()
        prefill_engine.load_state_dict(checkpoint["prefill_engine"])
        step_engine = _new_trt_module().cuda()
        step_engine.load_state_dict(checkpoint["decoder_step_engine"])

        text_state = checkpoint["text_decoder_extra_state"]
        token_embedding = nn.Embedding(dims.n_vocab, dims.n_text_state)
        token_embedding.load_state_dict(text_state["token_embedding"])
        positional_embedding = nn.Parameter(text_state["positional_embedding"]).cuda()
        ln_layer = LayerNorm(dims.n_text_state)
        ln_layer.load_state_dict(text_state["ln"])
        mask = text_state["mask"]
        if not mask.is_cuda:
            mask = mask.cuda()

        return TextDecoderTRTKV(
            DecoderEngines(
                cross_kv=cross_kv_engine,
                prefill=prefill_engine,
                step=step_engine,
            ),
            TextDecoderState(
                token_embedding=token_embedding,
                positional_embedding=positional_embedding,
                ln=ln_layer,
                mask=mask,
            ),
            dims,
        )

    @classmethod
    @torch.no_grad()
    def load(cls, trt_model_path: str) -> WhisperTRT:
        """Load a TensorRT checkpoint from disk into a ready-to-run model."""
        checkpoint = torch.load(trt_model_path)
        dims = ModelDimensions(**checkpoint["dims"])
        encoder = cls._load_audio_encoder(checkpoint)
        decoder = cls._load_text_decoder(checkpoint, dims)

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


# Bump when the on-disk engine layout changes so stale checkpoints are
# rebuilt rather than mis-loaded. "kv3" = the three-engine KV-cached decoder
# (cross_kv_engine + prefill_engine + decoder_step_engine). Pre-KV engines
# used no schema tag, so their files remain on disk untouched and a code
# revert reuses them.
_ENGINE_SCHEMA = "kv4"


def get_model_filename(name: str, quant_mode: str) -> str:
    """
    Returns the compute-type-aware filename for a given model and quantization mode.

    Each distinct compute type (float32, float16, int8) produces a separate cached
    engine file, preventing silent reuse of an engine built under a different precision.
    The engine-schema tag additionally invalidates caches whose serialized layout
    no longer matches the loader.

    Args:
        name (str): The model name (e.g. "tiny", "base.en").
        quant_mode (str): The quantization mode ("float32", "float16", or "int8").

    Returns:
        str: Filename with the quant mode and schema embedded
            (e.g. "tiny_trt_float16_kv4.pth").

    Raises:
        RuntimeError: If ``name`` is not a recognised model name.
    """
    if name not in MODEL_FILENAMES:
        raise RuntimeError(f"Model '{name}' is not supported by WhisperTRT.")
    base = MODEL_FILENAMES[name]
    stem, ext = os.path.splitext(base)
    return f"{stem}_{quant_mode}_{_ENGINE_SCHEMA}{ext}"


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
    # determine on-disk path — include quant_mode in filename to avoid silent
    # reuse of an engine built under a different precision.

    if path is None:
        filename = get_model_filename(name, WhisperTRTBuilder.get_compute_type())
        path = os.path.join(get_cache_dir(), filename)
        make_cache_dir()

    builder = MODEL_BUILDERS[name]
    if not os.path.exists(path):
        if not build:
            raise RuntimeError(f"No model found at {path}; pass build=True.")
        builder.build(path, verbose=verbose)

    trt_model = builder.load(path)

    try:
        silence = np.zeros((whisper.audio.N_SAMPLES,), dtype=np.float32)
        _ = trt_model.transcribe(silence, language=language, stream=False)
    except (RuntimeError, ValueError) as err:
        logger.debug("Warm-up skipped: %s", err)

    return trt_model
