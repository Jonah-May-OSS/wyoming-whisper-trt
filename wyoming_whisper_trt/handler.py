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


# Whisper's mel front-end is hard-wired for 16 kHz mono audio.
TARGET_RATE = 16000


def wav_bytes_to_np_array(wav_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes to 16 kHz mono float32, as Whisper expects.

    Mirrors the guarantees of ``whisper.audio.load_audio``: multi-channel
    audio is downmixed to mono and any sample rate is resampled to 16 kHz.
    Skipping either step feeds Whisper a malformed array — a multi-channel
    array in particular explodes ``log_mel_spectrogram``'s padding into a
    multi-gigabyte allocation.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_width = wf.getsampwidth()
        channels = wf.getnchannels()
        rate = wf.getframerate()
        raw_data = wf.readframes(wf.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sample_width)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {sample_width * 8}-bit")

    audio = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)

    # Normalize to [-1, 1]; 8-bit PCM is unsigned with a 128 offset.
    if dtype == np.uint8:
        audio = (audio - 128.0) / 128.0
    else:
        audio /= float(2 ** (8 * sample_width - 1))

    # Downmix to mono by averaging channels.
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    # Resample to 16 kHz via linear interpolation.
    if rate != TARGET_RATE and audio.size:
        target_len = round(audio.size * TARGET_RATE / rate)
        if target_len > 0:
            audio = np.interp(
                np.linspace(0.0, audio.size - 1, num=target_len),
                np.arange(audio.size),
                audio,
            ).astype(np.float32)

    return np.ascontiguousarray(audio, dtype=np.float32)


def _rms(audio: np.ndarray) -> float:
    """Root-mean-square amplitude of normalized ([-1, 1]) mono audio."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


@dataclass
class HandlerSettings:
    """Runtime settings for the event handler.

    Attributes:
        initial_prompt: Optional text to provide as context for the first
            transcription window. None means no initial prompt is provided.
        streaming: If True, emit intermediate transcription chunks in addition
            to the final text. Defaults to False.
        default_language: Default language code for transcription (e.g., "en",
            "es") or None for automatic language detection. Defaults to None.
        no_speech_threshold: Probability threshold (0.0-1.0) for suppressing
            Whisper's silence hallucinations (e.g. "www.mooji.org"). Drops a
            window whose ``<|nospeech|>`` probability meets or exceeds this
            threshold. None disables this feature. Defaults to 0.6.
        silence_rms_threshold: RMS energy threshold for the hard silence gate.
            Audio quieter than this normalized ([-1, 1]) root-mean-square value
            is short-circuited before reaching the model, emitting empty text.
            0.0 disables this feature. Defaults to 0.0.
    """

    initial_prompt: str | None = None
    streaming: bool = False
    default_language: str | None = None
    no_speech_threshold: float | None = 0.6
    silence_rms_threshold: float = 0.0


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

        # Cheap energy gate: near-silent audio only ever produces Whisper
        # hallucinations, so short-circuit it before touching the model. The
        # no-speech gate inside transcribe() is the accurate check; this is an
        # optional hard cutoff, disabled by default (threshold 0.0).
        threshold = self._settings.silence_rms_threshold
        if threshold > 0.0 and _rms(audio_np) < threshold:
            logger.debug("Audio below silence RMS threshold; emitting empty transcript")
            await self._emit_transcript("")
            self._wav_buffer = io.BytesIO()
            return

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
                        no_speech_threshold=self._settings.no_speech_threshold,
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

        await self._emit_transcript(final_text)

        # reset for next utterance

        self._wav_buffer = io.BytesIO()

    async def _emit_transcript(self, final_text: str) -> None:
        """Emit a completed transcript, matching the client's streaming mode.

        For streaming clients: TranscriptStart → TranscriptChunk → Transcript →
        TranscriptStop. For non-streaming clients: Transcript only. (HA's
        wyoming/stt.py ignores the streaming events and waits for Transcript.)
        """
        if self._settings.streaming:
            await self.write_event(TranscriptStart(language=self._language).event())
            if final_text:
                await self.write_event(TranscriptChunk(text=final_text).event())
            await self.write_event(Transcript(text=final_text).event())
            await self.write_event(TranscriptStop().event())
        else:
            await self.write_event(Transcript(text=final_text).event())

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
