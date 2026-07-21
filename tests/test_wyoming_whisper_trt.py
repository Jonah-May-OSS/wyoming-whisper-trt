"""End-to-end test for the wyoming_whisper_trt server.

Speaks the Wyoming protocol to the real server over TCP (the deployment
transport) and checks a known transcription. Requires a CUDA GPU
(TensorRT engine inference), so it skips on CPU-only CI runners; run it
on GPU hardware via script/test. With cold engine caches the first run
also builds the TensorRT engines, which can take a few minutes per
compute type.
"""

import asyncio
import io
import re
import socket
import sys
import wave
from asyncio.subprocess import DEVNULL, PIPE, Process
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart, AudioStop, wav_to_chunks
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe, Info

from wyoming_whisper_trt.handler import TARGET_RATE, _rms, wav_bytes_to_np_array

_CUDA_AVAILABLE = torch.cuda.is_available()

_DIR = Path(__file__).parent
_PROGRAM_DIR = _DIR.parent
_LOCAL_DIR = _PROGRAM_DIR / "local"
_SAMPLES_PER_CHUNK = 1024
# Generous: covers a cold TensorRT engine build before the port opens.
_STARTUP_TIMEOUT = 600
_TRANSCRIBE_TIMEOUT = 120

_INT8_WARNING_FRAGMENT = "int8 requests TensorRT implicit INT8"


def _free_port() -> int:
    """Ask the OS for a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _drain(stream: asyncio.StreamReader, buf: bytearray) -> None:
    """Continuously read a stream into a buffer so the pipe never fills."""
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return
        buf.extend(chunk)


def _server_diagnostics(
    proc: Process, stdout_buf: bytearray, stderr_buf: bytearray
) -> str:
    """Describe the server process state for a failure message."""
    return (
        f"server returncode={proc.returncode}\n"
        f"--- server stdout tail ---\n{stdout_buf.decode(errors='replace')[-2000:]}\n"
        f"--- server stderr tail ---\n{stderr_buf.decode(errors='replace')[-4000:]}"
    )


async def _connect_when_ready(
    proc: Process, port: int, stdout_buf: bytearray, stderr_buf: bytearray
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Poll the server port until it accepts, failing fast if the server dies."""
    deadline = asyncio.get_running_loop().time() + _STARTUP_TIMEOUT
    while True:
        if proc.returncode is not None:
            pytest.fail(
                "Server exited before accepting connections.\n"
                + _server_diagnostics(proc, stdout_buf, stderr_buf)
            )
        try:
            return await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            if asyncio.get_running_loop().time() > deadline:
                pytest.fail(
                    "Server did not open its port in time.\n"
                    + _server_diagnostics(proc, stdout_buf, stderr_buf)
                )
            await asyncio.sleep(0.5)


async def _next_event_of(
    reader: asyncio.StreamReader,
    is_type: Callable[[Any], bool],
    timeout: float | None,
) -> Any:
    """Read events until one matches the given Wyoming type predicate."""
    while True:
        event = await asyncio.wait_for(async_read_event(reader), timeout=timeout)
        if event is None:
            pytest.fail("Server closed the connection mid-conversation")
        if is_type(event.type):
            return event


def _make_wav_bytes(samples: np.ndarray, rate: int, channels: int) -> bytes:
    """Encode int16 PCM samples (interleaved if multi-channel) as WAV bytes."""
    buf = io.BytesIO()
    writer: wave.Wave_write = wave.open(buf, "wb")
    try:
        writer.setframerate(rate)
        writer.setsampwidth(2)
        writer.setnchannels(channels)
        writer.writeframes(samples.astype("<i2").tobytes())
    finally:
        writer.close()
    return buf.getvalue()


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("rate", [16000, 44100])
def test_wav_bytes_to_np_array_normalizes(channels: int, rate: int) -> None:
    """Any channel count / rate must decode to 1-D 16 kHz mono float32.

    Regression guard for the multi-channel array that exploded
    log_mel_spectrogram's padding into a multi-gigabyte allocation.
    """
    frames = rate  # one second
    mono = (np.sin(np.linspace(0, 220.0, frames)) * 10000).astype("<i2")
    interleaved = np.repeat(mono, channels) if channels > 1 else mono

    audio = wav_bytes_to_np_array(_make_wav_bytes(interleaved, rate, channels))

    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert audio.flags["C_CONTIGUOUS"]
    assert abs(audio.size - TARGET_RATE) <= 1  # resampled to ~1 s at 16 kHz
    assert audio.max() <= 1.0 and audio.min() >= -1.0


def test_rms_of_silence_is_zero() -> None:
    """Digital silence (and an empty array) must have zero RMS."""
    assert _rms(np.zeros(16000, dtype=np.float32)) == 0.0
    assert _rms(np.array([], dtype=np.float32)) == 0.0


def test_rms_of_full_scale_square_wave_is_one() -> None:
    """A full-scale ±1 signal has unit RMS; a quiet signal stays below a gate."""
    square = np.tile([1.0, -1.0], 8000).astype(np.float32)
    assert _rms(square) == pytest.approx(1.0)
    assert _rms(np.full(16000, 0.001, dtype=np.float32)) < 0.01


def test_is_no_speech_gate() -> None:
    """The no-speech gate fires only above threshold and stays off when disabled."""
    tensorrt = pytest.importorskip("tensorrt")  # noqa: F841  # model.py needs TRT
    from whisper_trt.model import WhisperTRT

    class _Tok:
        no_speech = 2

    # Logits over a tiny vocab; index 2 is the <|nospeech|> token.
    high = torch.tensor([[[0.0, 0.0, 10.0, 0.0]]])  # softmax ~0.9995 on no_speech
    low = torch.tensor([[[10.0, 0.0, 0.0, 0.0]]])  # near-zero on no_speech

    gate = WhisperTRT._is_no_speech
    assert gate(None, _Tok(), high, 0.6) is True
    assert gate(None, _Tok(), low, 0.6) is False
    assert gate(None, _Tok(), high, None) is False  # disabled


@pytest.mark.asyncio
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires a CUDA GPU")
@pytest.mark.parametrize(
    ("compute_type", "decoder_mode"),
    [("float16", "kv"), ("int8", "kv"), ("float16", "simple")],
)
async def test_wyoming_whisper_trt(compute_type: str, decoder_mode: str) -> None:
    """Transcribe a known WAV through the real server over TCP.

    Covers both decoder modes: the KV-cached three-engine decoder and the
    single-engine "simple" decoder.
    """
    port = _free_port()
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "wyoming_whisper_trt",
        "--uri",
        f"tcp://127.0.0.1:{port}",
        "--model",
        "base",
        "--data-dir",
        str(_LOCAL_DIR),
        "--compute-type",
        compute_type,
        "--decoder-mode",
        decoder_mode,
        "--language",
        "en",
        stdin=DEVNULL,
        stdout=PIPE,
        stderr=PIPE,
        cwd=_PROGRAM_DIR,
    )
    assert proc.stdout is not None and proc.stderr is not None

    # Drain both streams in the background: logs and native TensorRT output
    # can exceed the pipe buffer and would otherwise deadlock the server.
    stdout_buf = bytearray()
    stderr_buf = bytearray()
    drains = [
        asyncio.create_task(_drain(proc.stdout, stdout_buf)),
        asyncio.create_task(_drain(proc.stderr, stderr_buf)),
    ]

    writer = None
    try:
        reader, writer = await _connect_when_ready(proc, port, stdout_buf, stderr_buf)

        # Protocol sanity: the server must describe at least one ASR model.
        await async_write_event(Describe().event(), writer)
        info = Info.from_event(
            await _next_event_of(reader, Info.is_type, _TRANSCRIBE_TIMEOUT)
        )
        assert len(info.asr) == 1, "Expected one asr service"
        assert len(info.asr[0].models) > 0, "Expected at least one model"

        # Transcribe a known WAV.
        await async_write_event(Transcribe(language="en").event(), writer)
        wav_path = _DIR / "turn_on_the_living_room_lamp.wav"
        with wave.open(str(wav_path), "rb") as wav:
            await async_write_event(
                AudioStart(
                    rate=wav.getframerate(),
                    width=wav.getsampwidth(),
                    channels=wav.getnchannels(),
                ).event(),
                writer,
            )
            for chunk in wav_to_chunks(wav, _SAMPLES_PER_CHUNK):
                await async_write_event(chunk.event(), writer)
            await async_write_event(AudioStop().event(), writer)

        transcript = Transcript.from_event(
            await _next_event_of(reader, Transcript.is_type, _TRANSCRIBE_TIMEOUT)
        )
        text = re.sub(r"[^a-z ]", "", transcript.text.lower().strip())
        assert text == "turn on the living room lamp"
    finally:
        if writer is not None:
            writer.close()
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=30)
            except TimeoutError:
                proc.kill()
                await proc.wait()
        await asyncio.gather(*drains)

    if compute_type == "int8":
        # The int8 no-op warning added after benchmarking must be present.
        assert _INT8_WARNING_FRAGMENT in stderr_buf.decode(errors="replace"), (
            "Expected the int8 reality-check warning in server logs"
        )
