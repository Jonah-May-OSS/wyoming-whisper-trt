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

import os
import re
import time
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch2trt
import tensorrt

from whisper import load_model
from whisper.model import LayerNorm, Tensor, ModelDimensions
from whisper.tokenizer import Tokenizer, TO_LANGUAGE_CODE
import whisper.audio
from dataclasses import asdict

from .cache import get_cache_dir, make_cache_dir
from .__version__ import __version__

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# -----------------------------------------------------------------------------
# AUDIO ENCODER
# -----------------------------------------------------------------------------


class _AudioEncoderEngine(nn.Module):
    """
    Audio Encoder Engine for Whisper TRT.

    This module allows for online substitution of the positional embedding.
    (This implementation preserves the original logic.)
    """

    def __init__(
        self, conv1: nn.Conv1d, conv2: nn.Conv1d, blocks: nn.Module, ln_post: LayerNorm
    ) -> None:
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.blocks = blocks
        self.ln_post = ln_post

    @torch.no_grad()
    def forward(self, x: Tensor, positional_embedding: Tensor) -> Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = (x + positional_embedding).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        return x


class AudioEncoderTRT(nn.Module):
    """
    Audio Encoder using the TRT optimized engine.
    (This preserves the original method of slicing the positional embedding.)
    """

    def __init__(
        self, engine: torch2trt.TRTModule, positional_embedding: torch.Tensor
    ) -> None:
        super().__init__()
        self.engine = engine
        self.register_buffer("positional_embedding", positional_embedding)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        n_audio_ctx = int(x.shape[2] // 2)
        pos_embed = self.positional_embedding[-n_audio_ctx:, :]
        x = self.engine(x, pos_embed)
        return x


# -----------------------------------------------------------------------------
# TEXT DECODER
# -----------------------------------------------------------------------------


class _TextDecoderEngine(nn.Module):
    """
    Text Decoder Engine for Whisper TRT.
    """

    def __init__(self, blocks: nn.Module) -> None:
        super().__init__()
        self.blocks = blocks

    @torch.no_grad()
    def forward(self, x: Tensor, xa: Tensor, mask: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, xa, mask)
        return x


class TextDecoderTRT(nn.Module):
    """
    Text Decoder using the TRT optimized engine.
    """

    def __init__(
        self,
        engine: torch2trt.TRTModule,
        token_embedding: nn.Embedding,
        positional_embedding: nn.Parameter,
        ln: LayerNorm,
        mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.ln = ln
        self.register_buffer("mask", mask, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, xa: Tensor) -> Tensor:
        offset = 0
        token_emb = self.token_embedding(x).to(xa.device)
        pos_emb = self.positional_embedding[offset : offset + x.shape[-1]].to(xa.device)
        x = token_emb + pos_emb
        mask = self.mask.to(xa.device)
        x = self.engine(x, xa, mask)
        x = self.ln(x)
        weight = self.token_embedding.weight.to(x.device)
        logits = (x @ torch.transpose(weight, 0, 1)).float()
        return logits


# -----------------------------------------------------------------------------
# WHISPER TRT MODEL
# -----------------------------------------------------------------------------


class WhisperTRT(nn.Module):
    """
    Whisper model optimized with TensorRT.

    This implementation preserves the original transcription behavior.
    It supports multiple languages via the language parameter in transcribe().
    It uses a dedicated non-default CUDA stream.
    """

    def __init__(
        self,
        dims: ModelDimensions,
        encoder: AudioEncoderTRT,
        decoder: TextDecoderTRT,
        tokenizer: Optional[Tokenizer] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.stream = torch.cuda.Stream()
        # Do not enable TorchScript here.
        # self.forward = torch.jit.script(self.forward)

    def embed_audio(self, mel: Tensor) -> Tensor:
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: Tensor) -> Tensor:
        return self.decoder(tokens, audio_features)

    def forward(self, mel: Tensor, tokens: torch.Tensor) -> Tensor:
        return self.decoder(tokens, self.encoder(mel))

    @torch.no_grad()
    def transcribe(
        self,
        audio: str | np.ndarray,
        language: str = "auto",
        stream: bool = False,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        # If audio is a string, load it; if a NumPy array and not floating, convert.

        if isinstance(audio, str):
            audio = whisper.audio.load_audio(audio)
        elif isinstance(audio, np.ndarray):
            if not np.issubdtype(audio.dtype, np.floating):
                audio = audio.astype(np.float32) / 32768.0

        mel = whisper.audio.log_mel_spectrogram(audio, padding=whisper.audio.N_SAMPLES)[
            None, ...
        ].cuda()
        if mel.shape[2] > whisper.audio.N_FRAMES:
            mel = mel[:, :, : whisper.audio.N_FRAMES]
        load_time = time.perf_counter() - start_time

        chunks: List[str] = []
        max_len = self.dims.n_text_ctx + 1

        with torch.cuda.stream(self.stream):
            audio_features = self.embed_audio(mel)

            if self.tokenizer is not None:
                if language.lower() != "auto":
                    lang_code = language.lower()
                    if lang_code in TO_LANGUAGE_CODE:
                        lang_code = TO_LANGUAGE_CODE[lang_code]
                    self.tokenizer.language = lang_code
                    logger.debug("Tokenizer language set to: %s", lang_code)
                else:
                    self.tokenizer.language = None
                    logger.debug("Tokenizer set to auto language detection.")
            else:
                logger.warning("No tokenizer found; transcription may be degraded.")

            # Preallocate
            out_tokens = torch.empty((1, max_len), dtype=torch.long).cuda()
            out_tokens[0, 0] = self.tokenizer.sot
            cur_len = 1
            decode_start = time.perf_counter()

            for i in range(1, max_len):
                current_tokens = out_tokens[:, :cur_len]
                logits = self.logits(current_tokens, audio_features)
                next_token = logits.argmax(dim=-1)[:, -1]
                out_tokens[0, cur_len] = next_token
                cur_len += 1

                # if streaming mode, decode interim text so far
                if stream:
                    interim_tokens = out_tokens[
                        :, 2:cur_len
                    ]  # skip <sot> and <notimestamps>
                    interim_text = self._decode_tokens(interim_tokens)
                    chunks.append(interim_text)

                if next_token.item() == self.tokenizer.eot:
                    break

            # after loop, build final text
            final_tokens = out_tokens[:, 2 : cur_len - 1]
            final_text = self._decode_tokens(final_tokens)
            decode_time = time.perf_counter() - decode_start

        self.stream.synchronize()
        total_time = time.perf_counter() - start_time

        if self.verbose:
            logger.info(
                "Audio load & mel: %.1f ms, Decoding: %.1f ms, Total: %.1f ms",
                load_time * 1000,
                decode_time * 1000,
                total_time * 1000,
            )

        result = {"text": final_text}
        if stream:
            result["chunks"] = chunks
        return result

    @torch.no_grad()
    def transcribe_batch(
        self, audios: List[str | np.ndarray], language: str = "auto"
    ) -> List[Dict[str, str]]:
        start_time = time.perf_counter()
        mel_list = []
        for audio in audios:
            if isinstance(audio, str):
                audio = whisper.audio.load_audio(audio)
            elif isinstance(audio, np.ndarray):
                if not np.issubdtype(audio.dtype, np.floating):
                    audio = audio.astype(np.float32) / 32768.0
            audio_tensor = torch.tensor(audio, device="cpu").pin_memory()
            mel = whisper.audio.log_mel_spectrogram(
                audio_tensor.numpy(), padding=whisper.audio.N_SAMPLES
            )
            if mel.shape[1] > whisper.audio.N_FRAMES:
                mel = mel[:, : whisper.audio.N_FRAMES]
            mel_list.append(mel)
        mel_batch = np.stack(mel_list, axis=0)
        mel_batch = torch.tensor(mel_batch).cuda()
        load_time = time.perf_counter() - start_time

        with torch.cuda.stream(self.stream):
            audio_features = self.embed_audio(mel_batch)
            if self.tokenizer is not None:
                if language.lower() != "auto":
                    lang_code = language.lower()
                    if lang_code in TO_LANGUAGE_CODE:
                        lang_code = TO_LANGUAGE_CODE[lang_code]
                    self.tokenizer.language = lang_code
                    logger.debug("Tokenizer language set to: %s", lang_code)
                else:
                    self.tokenizer.language = None
                    logger.debug("Tokenizer set to auto language detection.")
            else:
                logger.warning("No tokenizer found; transcription may be degraded.")
            batch_size = mel_batch.shape[0]
            max_len = self.dims.n_text_ctx + 1
            out_tokens = torch.empty((batch_size, max_len), dtype=torch.long).cuda()
            out_tokens[:, 0] = self.tokenizer.sot
            cur_len = 1
            decode_start = time.perf_counter()
            for i in range(1, self.dims.n_text_ctx + 1):
                current_tokens = out_tokens[:, :cur_len]
                logits = self.logits(current_tokens, audio_features)
                next_tokens = logits.argmax(dim=-1)[:, -1]
                out_tokens[:, cur_len] = next_tokens
                cur_len += 1
                if (next_tokens == self.tokenizer.eot).all():
                    break
            tokens = out_tokens[:, 2 : cur_len - 1]
            texts = []
            for i in range(batch_size):
                t = self.tokenizer.decode(list(tokens[i].cpu().numpy()))
                t = re.sub(r"<\|transcribe\|><\|notimestamps\|>", "", t).strip()
                texts.append({"text": t})
            decode_time = time.perf_counter() - decode_start
        self.stream.synchronize()
        total_time = time.perf_counter() - start_time
        if self.verbose:
            logger.info(
                "Batched load & mel: %.1f ms, Batch decoding: %.1f ms, Total: %.1f ms",
                load_time * 1000,
                decode_time * 1000,
                total_time * 1000,
            )
        return texts

    @torch.no_grad()
    def get_supported_languages(self) -> List[str]:
        """
        Returns the list of supported languages. If the attached tokenizer has the
        attribute 'all_language_codes', that list is returned. Otherwise, defaults to ['en'].
        """
        if self.tokenizer is not None and hasattr(self.tokenizer, "all_language_codes"):
            return list(self.tokenizer.all_language_codes)
        return ["en"]

    def _decode_tokens(self, tokens: torch.Tensor) -> str:
        """Decode tokens to text, removing special markers."""
        text = self.tokenizer.decode(list(tokens.flatten().cpu().numpy()))
        return re.sub(r"<\|transcribe\|><\|notimestamps\|>", "", text).strip()


# -----------------------------------------------------------------------------
# BUILDER CLASSES FOR MULTILINGUAL AND ENGLISH-ONLY MODELS
# -----------------------------------------------------------------------------


class WhisperTRTBuilder:
    model: str
    fp16_mode: bool = True
    max_workspace_size: int = 1 << 30
    verbose: bool = False
    quant_mode: str = "fp16"  # Option: "fp16" or "int8"
    _tokenizer: Optional[Tokenizer] = None
    _dims: Optional[ModelDimensions] = None

    @classmethod
    @torch.no_grad()
    def _load_model_once(cls) -> ModelDimensions:
        if cls._dims is None:
            model_inst = load_model(cls.model).cuda().eval()
            cls._dims = model_inst.dims
        return cls._dims

    @classmethod
    @torch.no_grad()
    def build_text_decoder_engine(cls) -> torch2trt.TRTModule:
        dims = cls._load_model_once()
        model_inst = load_model(cls.model).cuda().eval()
        decoder_blocks_module = _TextDecoderEngine(model_inst.decoder.blocks)
        x = torch.randn(1, 1, dims.n_text_state).cuda()
        xa = torch.randn(1, dims.n_audio_ctx, dims.n_audio_state).cuda()
        mask = torch.randn(dims.n_text_ctx, dims.n_text_ctx).cuda()
        int8_mode = cls.quant_mode == "int8"
        engine = torch2trt.torch2trt(
            decoder_blocks_module,
            [x, xa, mask],
            use_onnx=True,
            int8_mode=int8_mode,
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
            fp16_mode=cls.fp16_mode if not int8_mode else False,
            log_level=tensorrt.Logger.VERBOSE if cls.verbose else tensorrt.Logger.ERROR,
        )
        return engine

    @classmethod
    @torch.no_grad()
    def build_audio_encoder_engine(cls) -> torch2trt.TRTModule:
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
        positional_embedding = model_inst.encoder.positional_embedding.cuda().detach()
        int8_mode = cls.quant_mode == "int8"
        engine = torch2trt.torch2trt(
            encoder_module,
            [x, positional_embedding],
            use_onnx=True,
            int8_mode=int8_mode,
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
            fp16_mode=cls.fp16_mode if not int8_mode else False,
            log_level=tensorrt.Logger.VERBOSE if cls.verbose else tensorrt.Logger.ERROR,
        )
        return engine

    @classmethod
    @torch.no_grad()
    def get_text_decoder_extra_state(cls) -> Dict[str, Any]:
        model_inst = load_model(cls.model).cuda().eval()
        extra_state = {
            "token_embedding": model_inst.decoder.token_embedding.state_dict(),
            "positional_embedding": model_inst.decoder.positional_embedding,
            "ln": model_inst.decoder.ln.state_dict(),
            "mask": model_inst.decoder.mask,
        }
        return extra_state

    @classmethod
    @torch.no_grad()
    def get_audio_encoder_extra_state(cls) -> Dict[str, Any]:
        model_inst = load_model(cls.model).cuda().eval()
        extra_state = {"positional_embedding": model_inst.encoder.positional_embedding}
        return extra_state

    @classmethod
    @torch.no_grad()
    def build(cls, output_path: str, verbose: bool = False) -> None:
        cls.verbose = verbose
        dims = asdict(load_model(cls.model).dims)
        checkpoint = {
            "whisper_trt_version": __version__,
            "dims": dims,
            "text_decoder_engine": cls.build_text_decoder_engine().state_dict(),
            "text_decoder_extra_state": cls.get_text_decoder_extra_state(),
            "audio_encoder_engine": cls.build_audio_encoder_engine().state_dict(),
            "audio_encoder_extra_state": cls.get_audio_encoder_extra_state(),
        }
        torch.save(checkpoint, output_path)

    @classmethod
    def get_tokenizer(cls) -> Tokenizer:
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
    @torch.no_grad()
    def load(cls, trt_model_path: str) -> WhisperTRT:
        checkpoint = torch.load(trt_model_path)
        dims = ModelDimensions(**checkpoint["dims"])
        # Audio encoder.

        audio_encoder_engine = torch2trt.TRTModule().cuda()
        audio_encoder_engine.load_state_dict(checkpoint["audio_encoder_engine"])
        aes = checkpoint["audio_encoder_extra_state"]
        audio_positional_embedding = aes["positional_embedding"]
        encoder = AudioEncoderTRT(audio_encoder_engine, audio_positional_embedding)
        # Text decoder.

        text_decoder_engine = torch2trt.TRTModule().cuda()
        text_decoder_engine.load_state_dict(checkpoint["text_decoder_engine"])
        tes = checkpoint["text_decoder_extra_state"]
        text_token_embedding = nn.Embedding(dims.n_vocab, dims.n_text_state)
        text_token_embedding.load_state_dict(tes["token_embedding"])
        text_positional_embedding = nn.Parameter(tes["positional_embedding"]).cuda()
        text_ln = LayerNorm(dims.n_text_state)
        text_ln.load_state_dict(tes["ln"])
        text_mask = tes["mask"]
        if not text_mask.is_cuda:
            text_mask = text_mask.cuda()
        decoder = TextDecoderTRT(
            text_decoder_engine,
            text_token_embedding,
            text_positional_embedding,
            text_ln,
            text_mask,
        )
        whisper_trt = WhisperTRT(
            dims, encoder, decoder, cls.get_tokenizer(), verbose=cls.verbose
        )
        whisper_trt = whisper_trt.cuda().eval()
        return whisper_trt


# -----------------------------------------------------------------------------
# ENGLISH-ONLY MODEL BUILDERS
# -----------------------------------------------------------------------------


class EnBuilder(WhisperTRTBuilder):
    @classmethod
    def get_tokenizer(cls) -> Tokenizer:
        tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=False, num_languages=99, language="en", task="transcribe"
        )
        return tokenizer


class TinyEnBuilder(EnBuilder):
    model: str = "tiny.en"


class BaseEnBuilder(EnBuilder):
    model: str = "base.en"


class SmallEnBuilder(EnBuilder):
    model: str = "small.en"


# -----------------------------------------------------------------------------
# MULTILINGUAL MODEL BUILDERS
# -----------------------------------------------------------------------------


class TinyBuilder(WhisperTRTBuilder):
    model: str = "tiny"


class BaseBuilder(WhisperTRTBuilder):
    model: str = "base"


class SmallBuilder(WhisperTRTBuilder):
    model: str = "small"


class MediumBuilder(WhisperTRTBuilder):
    model: str = "medium"


class LargeBuilder(WhisperTRTBuilder):
    model: str = "large"


class LargeV2Builder(WhisperTRTBuilder):
    model: str = "large-v2"


class LargeV3Builder(WhisperTRTBuilder):
    model: str = "large-v3"


class LargeV3TurboBuilder(WhisperTRTBuilder):
    model: str = "large-v3-turbo"


# -----------------------------------------------------------------------------
# MODEL FILE-NAMING & BUILDER DICTIONARIES
# -----------------------------------------------------------------------------


MODEL_FILENAMES = {
    # English-only models:
    "tiny.en": "tiny_en_trt.pth",
    "base.en": "base_en_trt.pth",
    "small.en": "small_en_trt.pth",
    # Multilingual models:
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
    # English-only models:
    "tiny.en": TinyEnBuilder,
    "base.en": BaseEnBuilder,
    "small.en": SmallEnBuilder,
    # Multilingual models:
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
    name: str, path: Optional[str] = None, build: bool = True, verbose: bool = False
) -> WhisperTRT:
    # print current precision settings
    logger.debug(
        "Loading TRT model '%s' with quant_mode=%s, fp16_mode=%s",
        name,
        WhisperTRTBuilder.quant_mode,
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
            raise RuntimeError(
                f"No model found at {path}. Please call load_trt_model with build=True."
            )
        else:
            builder.build(path, verbose=verbose)
    return builder.load(path)
