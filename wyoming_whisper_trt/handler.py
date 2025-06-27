"""Event handler for clients of the server."""

import argparse
import asyncio
import io
import logging
import time
import wave
from typing import Optional

import numpy as np
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
        # how many raw bytes before we do the next interim decode
        # 0.25 s of 16 kHz, 16-bit stereo -> 0.25 * 16k * 2 * 2 = 16 000 bytes
        self._partial_threshold = 16000

        # buffer raw PCM so we can re-run only the new audio
        self._pcm_buffer = bytearray()

        # have we already sent transcript-start?
        self._sent_start = False

        # how many interim chunks we’ve already emitted
        self._last_sent_chunk = 0

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
        """Process incoming audio chunks and stream back transcription."""
        chunk = AudioChunk.from_event(event)

        # Initialize WAV writer if necessary
        if self._wave_writer is None:
            self._wav_buffer = io.BytesIO()
            self._wave_writer = wave.open(self._wav_buffer, "wb")
            self._wave_writer.setframerate(chunk.rate)
            self._wave_writer.setsampwidth(chunk.width)
            self._wave_writer.setnchannels(chunk.channels)
            self._sample_width = chunk.width
            self._channels = chunk.channels
            logger.debug(
                f"Initialized WAV buffer with {self._partial_threshold} bytes threshold"
            )

        # Write to WAV buffer and accumulate PCM
        self._wave_writer.writeframes(chunk.audio)
        self._pcm_buffer.extend(chunk.audio)

        # Emit TranscriptStart if this is the first chunk
        if not self._sent_start:
            await self.write_event(TranscriptStart(language=self._language).event())
            logger.debug("➡️  Emitted TranscriptStart")
            self._sent_start = True

        # Sliding window decode
        window = self._partial_threshold
        hop = window // 2  # 50% overlap

        # Limit the size of the PCM buffer to avoid excessive memory use
        # Only trim if we have more than a reasonable amount of processed data
        max_buffer_size = self._partial_threshold * 10
        if len(self._pcm_buffer) > max_buffer_size:
            # Keep at least 2 windows worth of data to maintain overlap
            keep_size = window * 2
            if len(self._pcm_buffer) > keep_size:
                trim_to = len(self._pcm_buffer) - keep_size
                del self._pcm_buffer[:trim_to]
                logger.debug(f"Trimmed PCM buffer, keeping {keep_size} bytes")

        # Process audio chunks with sliding window
        while len(self._pcm_buffer) >= window:
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[self._sample_width]
            pcm = np.frombuffer(self._pcm_buffer[:window], dtype=dtype)
            if self._channels > 1:
                pcm = pcm.reshape(-1, self._channels)

            # Normalize to float32
            if pcm.dtype != np.float32:
                pcm = pcm.astype(np.float32) / float(2 ** (8 * self._sample_width - 1))

            # Transcribe the chunk in a thread-safe manner
            loop = asyncio.get_event_loop()
            async with self.model_lock:
                result = await loop.run_in_executor(
                    None,
                    lambda pcm=pcm: self.model.transcribe(
                        pcm, self._language, stream=True
                    ),
                )

            # Emit each chunk's transcription result immediately
            chunks = result.get("chunks", [])
            for text in chunks[self._last_sent_chunk :]:
                await self.write_event(TranscriptChunk(text=text).event())
                logger.debug(f"➡️ Emitted TranscriptChunk: {text!r}")
            self._last_sent_chunk = len(chunks)

            # Drop just the hop amount to maintain overlap
            del self._pcm_buffer[:hop]

    async def _handle_audio_stop(self) -> None:
        """Handles AudioStop by emitting the final transcript and closing out the stream."""
        if self._wave_writer is None:
            logger.warning("AudioStop received but no audio was recorded.")
            return

        # 1) Close out your WAV writer (so your on-disk or in-memory file is valid again)
        try:
            self._wave_writer.close()
            logger.debug("Finalized in-memory WAV buffer.")
        except wave.Error as e:
            logger.error("Failed to finalize in-memory WAV buffer: %s", e)
            raise
        finally:
            self._wave_writer = None

        # 2) If you still have any buffered PCM that hasn't been sent in an interim chunk,
        #    do one last non-streaming pass so we can emit the final transcript.
        if self._pcm_buffer:
            wav_bytes = self._wav_buffer.getvalue()
            audio_np = wav_bytes_to_np_array(wav_bytes)
            async with self.model_lock:
                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: self.model.transcribe(
                            audio_np, self._language, stream=False
                        ),
                    )
                    final_text = result.get("text", "")
                    logger.debug("➡️  Emitting final Transcript: %r", final_text)
                    await self.write_event(Transcript(text=final_text).event())
                except Exception as e:
                    logger.error("Final transcription failed: %s", e, exc_info=True)
                    await self.write_event(
                        Transcript(text=f"Transcription failed: {str(e)[:100]}").event()
                    )

        # 3) Emit the Wyoming “end of stream” event
        logger.debug("➡️  Emitting TranscriptStop")
        await self.write_event(TranscriptStop().event())

        # 4) Reset everything for the next utterance
        self._pcm_buffer.clear()
        self._sent_start = False
        self._last_sent_chunk = 0

        # 5) Restore your language and clean up the WAV buffer
        self._language = (
            self.cli_args.language
            if hasattr(self.cli_args, "language")
            else self.default_language
        )
        logger.debug("Reset language to: %s", self._language)

        # 6) Clean up the old WAV buffer and replace it with a fresh one
        self._wav_buffer.close()
        self._wav_buffer = io.BytesIO()

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
