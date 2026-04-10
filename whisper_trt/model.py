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

import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import tensorrt
import torch
import whisper.audio
from torch import nn
from whisper import load_model
from whisper.model import LayerNorm, ModelDimensions, Tensor
from whisper.tokenizer import TO_LANGUAGE_CODE, Tokenizer

import torch2trt

from .__version__ import __version__
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


@dataclass
class TextDecoderState:
    """State needed by TextDecoderTRT on top of the TensorRT engine."""

    token_embedding: nn.Embedding
    positional_embedding: torch.Tensor
    ln: LayerNorm
    mask: torch.Tensor


@dataclass
class WhisperTRTConfig:
    """Optional runtime configuration for the WhisperTRT model."""

    tokenizer: Tokenizer | None = None
    verbose: bool = False


@dataclass
class DecodeRequest:
    """Inputs required for one autoregressive decode pass."""

    out_tokens: torch.Tensor
    cur_len: int
    prompt_len: int
    max_len: int
    audio_features: Tensor
    stream: bool


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


class _TextDecoderEngine(nn.Module):
    """Torch module form of the Whisper decoder blocks for TRT conversion."""

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
    """Whisper text decoder that runs through a TensorRT engine."""

    def __init__(self, engine: Any, state: TextDecoderState) -> None:
        super().__init__()
        self.engine = engine
        self.token_embedding = state.token_embedding
        self.positional_embedding = state.positional_embedding
        self.ln = state.ln
        self.register_buffer("mask", state.mask, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, xa: Tensor) -> Tensor:
        """Decode token ids into next-token logits."""
        token_emb = self.token_embedding(x).to(xa.device)
        pos_emb = self.positional_embedding[: x.shape[-1]].to(xa.device)
        hidden = token_emb + pos_emb
        hidden = self.engine(hidden, xa, self.mask.to(xa.device))
        hidden = self.ln(hidden)
        weight = self.token_embedding.weight.to(hidden.device)
        return (hidden @ torch.transpose(weight, 0, 1)).float()

    def summary(self) -> str:
        """Return a short human-readable component summary."""
        return "TensorRT text decoder wrapper"


class WhisperTRT(nn.Module):
    """Whisper model optimized for TensorRT inference."""

    def __init__(
        self,
        dims: ModelDimensions,
        encoder: AudioEncoderTRT,
        decoder: TextDecoderTRT,
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
        """Return logits for token ids conditioned on audio features."""
        return self.decoder(tokens, audio_features)

    def forward(self, mel: Tensor, tokens: torch.Tensor) -> Tensor:
        """Run end-to-end forward pass from mel input to logits."""
        return self.decoder(tokens, self.encoder(mel))

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

    def _decode_sequence(
        self,
        tokenizer: Tokenizer,
        request: DecodeRequest,
    ) -> tuple[str, list[str], float]:
        """Autoregressively decode tokens and return final text and interim chunks."""
        chunks: list[str] = []
        decode_start = time.perf_counter()

        for _ in range(request.cur_len, request.max_len):
            token_logits = self.logits(
                request.out_tokens[:, : request.cur_len],
                request.audio_features,
            )
            next_token = token_logits.argmax(dim=-1)[0, -1]
            request.out_tokens[0, request.cur_len] = next_token
            request.cur_len += 1

            if request.stream:
                interim = request.out_tokens[:, request.prompt_len : request.cur_len]
                chunks.append(self._decode_tokens(interim))

            if next_token.item() == tokenizer.eot:
                break

        end_index = request.cur_len
        if request.out_tokens[0, request.cur_len - 1].item() == tokenizer.eot:
            end_index = request.cur_len - 1

        final_ids = request.out_tokens[:, request.prompt_len : end_index]
        final_text = self._decode_tokens(final_ids)
        decode_time = time.perf_counter() - decode_start
        return final_text, chunks, decode_time

    def _audio_to_mel(self, audio_array: np.ndarray) -> tuple[Tensor, float]:
        """Convert normalized audio samples to a mel spectrogram tensor."""
        load_start = time.perf_counter()
        mel = whisper.audio.log_mel_spectrogram(
            audio_array,
            padding=whisper.audio.N_SAMPLES,
        )[None, ...].cuda()
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

        torch.cuda.empty_cache()

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
    max_workspace_size: int = 1 << 30
    verbose: bool = False
    _tokenizer: Tokenizer | None = None
    _dims: ModelDimensions | None = None

    @classmethod
    @torch.no_grad()
    def _load_model_once(cls) -> ModelDimensions:
        """Load base Whisper model once per builder class and cache dimensions."""
        if cls._dims is None:
            cls._dims = load_model(cls.model).cuda().eval().dims
        return cls._dims

    @classmethod
    @torch.no_grad()
    def build_text_decoder_engine(cls) -> Any:
        """Build and return a TensorRT text decoder engine."""
        dims = cls._load_model_once()
        model_inst = load_model(cls.model).cuda().eval()
        decoder_blocks_module = _TextDecoderEngine(model_inst.decoder.blocks)
        x = torch.randn(1, 1, dims.n_text_state).cuda()
        xa = torch.randn(1, dims.n_audio_ctx, dims.n_audio_state).cuda()
        mask = torch.randn(dims.n_text_ctx, dims.n_text_ctx).cuda()
        return _torch2trt_convert(
            decoder_blocks_module,
            [x, xa, mask],
            use_onnx=True,
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
            max_workspace_size=cls.max_workspace_size,
            fp16_mode=cls.fp16_mode,
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
        return _torch2trt_convert(
            encoder_module,
            [x, positional_embedding],
            use_onnx=True,
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
            fp16_mode=cls.fp16_mode,
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
            "text_decoder_engine": cls.build_text_decoder_engine().state_dict(),
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
    ) -> TextDecoderTRT:
        """Construct TextDecoderTRT from a serialized checkpoint."""
        text_decoder_engine = _new_trt_module().cuda()
        text_decoder_engine.load_state_dict(checkpoint["text_decoder_engine"])

        text_state = checkpoint["text_decoder_extra_state"]
        token_embedding = nn.Embedding(dims.n_vocab, dims.n_text_state)
        token_embedding.load_state_dict(text_state["token_embedding"])
        positional_embedding = nn.Parameter(text_state["positional_embedding"]).cuda()
        ln_layer = LayerNorm(dims.n_text_state)
        ln_layer.load_state_dict(text_state["ln"])
        mask = text_state["mask"]
        if not mask.is_cuda:
            mask = mask.cuda()

        return TextDecoderTRT(
            text_decoder_engine,
            TextDecoderState(
                token_embedding=token_embedding,
                positional_embedding=positional_embedding,
                ln=ln_layer,
                mask=mask,
            ),
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


def load_trt_model(
    name: str,
    path: str | None = None,
    build: bool = True,
    verbose: bool = False,
    language: str = "auto",
) -> WhisperTRT:
    """Load (or build and then load) a TensorRT Whisper model by name."""
    logger.debug(
        "Loading TRT model '%s' with fp16_mode=%s",
        name,
        WhisperTRTBuilder.fp16_mode,
    )

    if name not in MODEL_BUILDERS:
        raise RuntimeError(f"Model '{name}' is not supported by WhisperTRT.")

    if path is None:
        path = os.path.join(get_cache_dir(), MODEL_FILENAMES[name])
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
