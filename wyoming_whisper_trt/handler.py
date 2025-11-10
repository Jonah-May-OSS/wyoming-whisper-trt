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


class NanosecondFormatter(logging.Formatter):
    """Custom formatter to include nanoseconds in log timestamps."""

    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        ct = record.created
        t = time.localtime(ct)
        s = time.strftime("%Y-%m-%d %H:%M:%S", t)
        return f"{s}.{int(ct * 1e9) % 1_000_000_000:09d}"


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
        streaming: bool = False,
        default_language: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        # remove unused kwargs

        super().__init__(reader, writer, *args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.initial_prompt = initial_prompt
        self.is_streaming = streaming
        self._language = default_language or getattr(self.cli_args, "language", None)

        # sliding-window threshold (bytes) for interim chunks

        self._partial_threshold = 16000
        self._pcm_buffer = bytearray()

        # WAV buffer for final flush

        self._wav_buffer = io.BytesIO()
        self._wave_writer: Optional[wave.Wave_write] = None
        self._sample_width: Optional[int] = None
        self._channels: Optional[int] = None

        # interim state

        self._sent_start = False
        self._last_sent_chunk = 0
        self._last_emitted_text = ""

        formatter = NanosecondFormatter("%(asctime)s [%(levelname)s] %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.handlers = [handler]

    async def handle_event(self, event: Event) -> bool:
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
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    async def _handle_audio_chunk(self, event: Event) -> None:
        chunk = AudioChunk.from_event(event)

        # init WAV writer if needed

        if self._wave_writer is None:
            self._wav_buffer = io.BytesIO()
            self._wave_writer = wave.open(self._wav_buffer, "wb")
            self._wave_writer.setframerate(chunk.rate)
            self._wave_writer.setsampwidth(chunk.width)
            self._wave_writer.setnchannels(chunk.channels)
            self._sample_width = chunk.width
            self._channels = chunk.channels
            logger.debug(
                "Initialized WAV buffer; threshold %d bytes", self._partial_threshold
            )
        # accumulate audio

        self._wave_writer.writeframes(chunk.audio)
        self._pcm_buffer.extend(chunk.audio)

        # sliding-window interim decoding only if streaming enabled

        if self.is_streaming:
            # emit TranscriptStart once

            if not self._sent_start:
                await self.write_event(TranscriptStart(language=self._language).event())
                logger.debug("➡️ Emitted TranscriptStart")
                self._sent_start = True
            window = self._partial_threshold
            hop = window // 2
            max_buf = window * 10
            if len(self._pcm_buffer) > max_buf:
                # Keep enough data for overlap + next processing

                keep_size = max(window * 3, len(self._pcm_buffer) - max_buf + window)
                del self._pcm_buffer[: len(self._pcm_buffer) - keep_size]
                logger.debug("Trimmed PCM buffer to %d bytes", len(self._pcm_buffer))
            # capture prompt once

            prompt = self.initial_prompt
            if prompt is not None:
                # ensure prompt is only used for the first decode

                self.initial_prompt = None
            while len(self._pcm_buffer) >= window:
                raw = self._pcm_buffer[:window]
                dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[self._sample_width]
                pcm = np.frombuffer(raw, dtype=dtype)
                if self._channels and self._channels > 1:
                    pcm = pcm.reshape(-1, self._channels)
                if pcm.dtype != np.float32:
                    pcm = pcm.astype(np.float32) / float(
                        2 ** (8 * self._sample_width - 1)
                    )
                loop = asyncio.get_event_loop()
                async with self.model_lock:
                    result = await loop.run_in_executor(
                        None,
                        lambda pcm=pcm, prompt=prompt: self.model.transcribe(
                            pcm,
                            self._language,
                            stream=True,
                            initial_prompt=prompt,
                        ),
                    )
                # emit any new chunks

                new_chunks = result.get("chunks", [])[self._last_sent_chunk :]
                if new_chunks:
                    for text in new_chunks:
                        clean = text.strip()
                        if clean and clean != self._last_emitted_text:
                            await self.write_event(TranscriptChunk(text=clean).event())
                            logger.debug("➡️ Emitted TranscriptChunk: %r", clean)
                            self._last_emitted_text = clean
                    self._last_sent_chunk = len(result.get("chunks", []))
                del self._pcm_buffer[:hop]

    async def _handle_audio_start(self) -> None:
        """Reset all buffers/state when a new audio stream begins."""
        logger.debug("Received AudioStart: resetting buffers/state")
        # clear out any leftover PCM data
        self._pcm_buffer.clear()

        # reset WAV writer
        self._wave_writer = None
        self._wav_buffer = io.BytesIO()
        self._sample_width = None
        self._channels = None

        # reset streaming flags/counters
        self._sent_start = False
        self._last_sent_chunk = 0
        self._last_emitted_text = ""

    async def _handle_audio_stop(self) -> None:
        """Handles AudioStop by emitting a final transcript and stopping streaming."""
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
        # prepare final prompt if not yet used (for non-streaming or first call)

        prompt = self.initial_prompt
        if prompt is not None:
            self.initial_prompt = None
        # run one non-streaming transcription for final text

        wav_bytes = self._wav_buffer.getvalue()
        audio_np = wav_bytes_to_np_array(wav_bytes)
        loop = asyncio.get_event_loop()
        async with self.model_lock:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: self.model.transcribe(
                        audio_np,
                        self._language,
                        stream=False,
                        initial_prompt=prompt,
                    ),
                )
                final_text = result.get("text", "").strip()
                logger.debug("➡️ Emitting final Transcript: %r", final_text)
                await self.write_event(Transcript(text=final_text).event())
            except Exception as e:
                logger.error("Final transcription failed: %s", e, exc_info=True)
                await self.write_event(
                    Transcript(text=f"Transcription failed: {str(e)[:100]}").event()
                )
        # emit TranscriptStop

        await self.write_event(TranscriptStop().event())

        # reset state for next utterance

        self._pcm_buffer.clear()
        self._sent_start = False
        self._last_sent_chunk = 0
        self._last_emitted_text = ""
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
        if self._wav_buffer:
            self._wav_buffer.close()
