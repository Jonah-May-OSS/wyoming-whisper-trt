"""Wyoming event handler for Whisper TRT speech transcription."""

import asyncio
import io
import logging
import wave
from dataclasses import dataclass

import numpy as np
from wyoming.asr import (
    Transcribe,
    Transcript,
    TranscriptChunk,
    TranscriptStart,
    TranscriptStop,
)
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

import whisper_trt

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def wav_bytes_to_np_array(wav_bytes: bytes) -> np.ndarray:
    """
    Read a WAV file from an in-memory bytes object and return a NumPy array of samples.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        raw_data = wf.readframes(wf.getnframes())
        sw = wf.getsampwidth()
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw)
        audio = np.frombuffer(raw_data, dtype=dtype)
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels())
        if not np.issubdtype(audio.dtype, np.floating):
            audio = audio.astype(np.float32) / float(2 ** (8 * sw - 1))
        return audio


@dataclass
class HandlerSettings:
    """Runtime settings for the event handler."""

    initial_prompt: str | None = None
    streaming: bool = False
    default_language: str | None = None


@dataclass
class HandlerContext:
    """Shared runtime objects required by each handler instance."""

    wyoming_info: Info
    model: whisper_trt.WhisperTRT
    model_lock: asyncio.Lock


class WhisperTrtEventHandler(AsyncEventHandler):
    """Event handler for clients utilizing the Whisper TRT model."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        context: HandlerContext,
        settings: HandlerSettings | None = None,
    ) -> None:
        """Initialize handler state for a single Wyoming client connection."""
        settings = settings or HandlerSettings()

        super().__init__(reader, writer)

        self.wyoming_info_event = context.wyoming_info.event()
        self.model = context.model
        self.model_lock = context.model_lock
        self._settings = settings
        self._language = settings.default_language

        # WAV buffer

        self._wav_buffer = io.BytesIO()
        self._wave_writer: wave.Wave_write | None = None

    async def handle_event(self, event: Event) -> bool:
        """Dispatch Wyoming protocol events for this connection."""
        logger.debug("Received event: %s", event.type)
        try:
            if Describe.is_type(event.type):
                await self._handle_describe()
                return True
            if AudioStart.is_type(event.type):
                await self._handle_audio_start()
                return True
            if AudioChunk.is_type(event.type):
                await self._handle_audio_chunk(event)
                return True
            if AudioStop.is_type(event.type):
                await self._handle_audio_stop()
                return False
            if Transcribe.is_type(event.type):
                await self._handle_transcribe(event)
                return True
        except (RuntimeError, OSError, ValueError, wave.Error) as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise
        return True

    async def _handle_audio_chunk(self, event: Event) -> None:
        """Append one chunk of PCM audio to the in-memory WAV buffer."""
        chunk = AudioChunk.from_event(event)

        # init WAV writer if needed

        if self._wave_writer is None:
            self._wav_buffer = io.BytesIO()
            self._wave_writer = wave.open(self._wav_buffer, "wb")
            self._wave_writer.setframerate(chunk.rate)
            self._wave_writer.setsampwidth(chunk.width)
            self._wave_writer.setnchannels(chunk.channels)
            logger.debug("Initialized WAV buffer.")

        # accumulate audio

        self._wave_writer.writeframes(chunk.audio)

    async def _handle_audio_start(self) -> None:
        """Reset all buffers/state when a new audio stream begins."""
        logger.debug("Received AudioStart: resetting buffers/state")

        # reset WAV writer
        self._wave_writer = None
        self._wav_buffer = io.BytesIO()

    async def _handle_audio_stop(self) -> None:
        """Handles AudioStop by emitting a final transcript."""
        if self._wave_writer is None:
            logger.warning("AudioStop received but no audio was recorded.")
            return

        # finalize WAV buffer

        try:
            self._wave_writer.close()
            logger.debug("Finalized in-memory WAV buffer.")
        except wave.Error as e:
            logger.error("Failed to finalize WAV buffer: %s", e)
            raise
        finally:
            self._wave_writer = None

        # prepare initial prompt

        prompt = self._settings.initial_prompt
        if prompt is not None:
            self._settings.initial_prompt = None

        # transcribe full audio

        wav_bytes = self._wav_buffer.getvalue()
        audio_np = wav_bytes_to_np_array(wav_bytes)
        loop = asyncio.get_event_loop()
        try:
            async with self.model_lock:
                result = await loop.run_in_executor(
                    None,
                    lambda: self.model.transcribe(
                        audio_np,
                        self._language or "auto",
                        stream=False,
                        initial_prompt=prompt,
                    ),
                )
            final_text = result.get("text", "").strip()
            logger.debug("➡️ Emitting final Transcript: %r", final_text)
        except (RuntimeError, OSError, ValueError) as err:
            logger.error("Transcription failed: %s", err, exc_info=True)
            await self.write_event(
                Transcript(text=f"Transcription failed: {str(err)[:100]}").event()
            )
            self._wav_buffer = io.BytesIO()
            return

        # for streaming clients: TranscriptStart → TranscriptChunk → Transcript → TranscriptStop
        # for non-streaming clients: Transcript only
        # (HA's wyoming/stt.py ignores streaming events and waits for Transcript)

        if self._settings.streaming:
            await self.write_event(TranscriptStart(language=self._language).event())
            if final_text:
                await self.write_event(TranscriptChunk(text=final_text).event())
            await self.write_event(Transcript(text=final_text).event())
            await self.write_event(TranscriptStop().event())
        else:
            await self.write_event(Transcript(text=final_text).event())

        # reset for next utterance

        self._wav_buffer = io.BytesIO()

    async def _handle_transcribe(self, event: Event) -> None:
        # allow client to change language on the fly

        tr = Transcribe.from_event(event)
        if tr.language:
            self._language = tr.language
            logger.debug("Language set to: %s", self._language)

    async def _handle_describe(self) -> None:
        # send the cached Info event

        await self.write_event(self.wyoming_info_event)

    def cleanup(self) -> None:
        """Release in-memory buffers owned by this handler."""
        if self._wav_buffer:
            self._wav_buffer.close()
