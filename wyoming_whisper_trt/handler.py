"""Event handler for clients of the server."""

import argparse
import asyncio
import io
import logging
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from wyoming.asr import (
    Transcribe,
    Transcript,
    TranscriptStart,
    TranscriptChunk,
    TranscriptStop,
)
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

import whisper_trt

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class NanosecondFormatter(logging.Formatter):
    """Custom formatter to include nanoseconds in log timestamps."""

    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        ct = record.created
        t = time.localtime(ct)
        s = time.strftime("%Y-%m-%d %H:%M:%S", t)
        return f"{s}.{int(ct * 1e9) % 1_000_000_000:09d}"


# Set up logging with the custom formatter.

formatter = NanosecondFormatter("%(asctime)s [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers = [handler]


def wav_bytes_to_np_array(wav_bytes: bytes) -> np.ndarray:
    """
    Read a WAV file from an in-memory bytes object and return a NumPy array of samples.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)
        sample_width = wf.getsampwidth()
        # Choose dtype based on sample width

        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError("Unsupported sample width")
        audio = np.frombuffer(raw_data, dtype=dtype)
        channels = wf.getnchannels()
        if channels > 1:
            audio = audio.reshape(-1, channels)
        return audio


class WhisperTrtEventHandler(AsyncEventHandler):
    """Event handler for clients utilizing the Whisper TRT model."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: whisper_trt.WhisperTRT,
        model_lock: asyncio.Lock,
        initial_prompt: Optional[str] = None,
        *args,
        model_is_lang_specific: bool = False,
        default_language: str = "auto",
        **kwargs,
    ) -> None:
        # Remove extra arguments so the base class won't receive them.

        kwargs.pop("model_is_lang_specific", None)
        kwargs.pop("default_language", None)
        super().__init__(reader, writer, *args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.initial_prompt = initial_prompt

        # Language and model-specific flags.

        self.model_is_lang_specific = model_is_lang_specific
        self.default_language = default_language

        self._language = (
            self.cli_args.language
            if hasattr(self.cli_args, "language")
            else self.default_language
        )

        # Use an in-memory buffer for WAV data.

        self._wav_buffer = io.BytesIO()
        self._wave_writer: Optional[wave.Wave_write] = None

    async def handle_event(self, event: Event) -> bool:
        """Handles incoming events from clients."""
        try:
            if AudioChunk.is_type(event.type):
                await self._handle_audio_chunk(event)
                return True
            if AudioStop.is_type(event.type):
                await self._handle_audio_stop()
                return False
            if Transcribe.is_type(event.type):
                await self._handle_transcribe(event)
                return True
            if Describe.is_type(event.type):
                await self._handle_describe()
                return True
            return True
        except Exception as e:
            logger.error("Error handling event %s: %s", event.type, e)
            return True

    async def _handle_audio_chunk(self, event: Event) -> None:
        """Handles an AudioChunk event by writing audio data to an in-memory WAV buffer."""
        chunk = AudioChunk.from_event(event)
        if self._wave_writer is None:
            try:
                # Initialize a new in-memory WAV writer.

                self._wav_buffer = io.BytesIO()
                self._wave_writer = wave.open(self._wav_buffer, "wb")
                self._wave_writer.setframerate(chunk.rate)
                self._wave_writer.setsampwidth(chunk.width)
                self._wave_writer.setnchannels(chunk.channels)
                logger.debug("Initialized in-memory WAV buffer.")
            except wave.Error as e:
                logger.error("Failed to initialize in-memory WAV buffer: %s", e)
                raise
        # Write the chunk's audio data.

        self._wave_writer.writeframes(chunk.audio)

    async def _handle_audio_stop(self) -> None:
        """Handles an AudioStop event by transcribing the recorded audio from memory,
        emitting Wyoming partial‐streaming events."""
        if self._wave_writer is None:
            logger.warning("AudioStop received but no audio was recorded.")
            return

        # Finalize the in‐memory WAV buffer
        try:
            self._wave_writer.close()
            logger.debug("Finalized in-memory WAV buffer.")
        except wave.Error as e:
            logger.error("Failed to finalize in-memory WAV buffer: %s", e)
            raise
        finally:
            self._wave_writer = None

        # 1) Emit transcript-start
        logger.debug("➡️  Emitting TranscriptStart")
        await self.write_event(TranscriptStart(language=self._language).event())

        # Get the WAV bytes and convert to numpy
        wav_bytes = self._wav_buffer.getvalue()
        audio_np = wav_bytes_to_np_array(wav_bytes)

        # 2) Do transcription under lock, potentially with streaming support
        async with self.model_lock:
            try:
                loop = asyncio.get_event_loop()

                # pass stream=True to get interim chunks if your model supports it
                result = await loop.run_in_executor(
                    None, lambda: self.model.transcribe(audio_np, self._language, True)
                )

                # 3) Emit each interim chunk
                for chunk_text in result.get("chunks", []):
                    logger.debug("➡️  Emitting TranscriptChunk: %r", chunk_text)
                    await self.write_event(TranscriptChunk(text=chunk_text).event())

                # 4) Emit backward‐compatible full transcript
                final_text = result.get("text", "")
                logger.debug("➡️  Emitting full Transcript: %r", final_text)
                await self.write_event(Transcript(text=final_text).event())

                logger.debug("Completed transcription request.")
            except Exception as e:
                logger.error("Transcription failed: %s", e)
                # still emit a stop so client won’t hang
                await self.write_event(Transcript(text="Transcription failed.").event())

        # 5) Emit transcript-stop
        logger.debug("➡️  Emitting TranscriptStop")
        await self.write_event(TranscriptStop().event())

        # Reset to default language and clean up
        self._language = (
            self.cli_args.language
            if hasattr(self.cli_args, "language")
            else self.default_language
        )
        logger.debug("Reset language to: %s", self._language)

        self._wav_buffer.close()

    async def _handle_transcribe(self, event: Event) -> None:
        """Handles a Transcribe event by setting the language."""
        transcribe = Transcribe.from_event(event)
        if transcribe.language:
            self._language = transcribe.language
            logger.debug("Language set to: %s", transcribe.language)

    async def _handle_describe(self) -> None:
        """Handles a Describe event by sending server information."""
        try:
            await self.write_event(self.wyoming_info_event)
            logger.debug("Sent server information.")
        except Exception as e:
            logger.error("Failed to send server information: %s", e)

    def cleanup(self) -> None:
        """Cleans up resources such as the in-memory buffer."""
        if self._wav_buffer:
            self._wav_buffer.close()
            logger.debug("Cleaned up in-memory WAV buffer.")
