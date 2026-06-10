"""End-to-end test for the wyoming_whisper_trt server.

Speaks the Wyoming protocol to the real server over stdio and checks a
known transcription. Requires a CUDA GPU (TensorRT engine inference), so
it skips on CPU-only CI runners; run it on GPU hardware via script/test.
With cold engine caches the first run also builds the TensorRT engines,
which can take a few minutes per compute type.
"""

import asyncio
import re
import sys
import wave
from asyncio.subprocess import PIPE
from pathlib import Path

import pytest
import torch
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart, AudioStop, wav_to_chunks
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe, Info

_CUDA_AVAILABLE = torch.cuda.is_available()

_DIR = Path(__file__).parent
_PROGRAM_DIR = _DIR.parent
_LOCAL_DIR = _PROGRAM_DIR / "local"
_SAMPLES_PER_CHUNK = 1024
# Generous: covers a cold TensorRT engine build on the first run.
_INFO_TIMEOUT = 600
_TRANSCRIBE_TIMEOUT = 120

_INT8_WARNING_FRAGMENT = "int8 requests TensorRT implicit INT8"


async def _next_event_of(is_type, reader, timeout):
    """Read events until one matches the given Wyoming type predicate."""
    while True:
        event = await asyncio.wait_for(async_read_event(reader), timeout=timeout)
        assert event is not None
        if is_type(event.type):
            return event


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="stdio transport is unreliable on Windows",
)
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires a CUDA GPU")
@pytest.mark.parametrize("compute_type", ["float16", "int8"])
async def test_wyoming_whisper_trt(compute_type: str) -> None:
    """Transcribe a known WAV through the real server."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "wyoming_whisper_trt",
        "--uri",
        "stdio://",
        "--model",
        "base",
        "--data-dir",
        str(_LOCAL_DIR),
        "--compute-type",
        compute_type,
        "--language",
        "en",
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        cwd=_PROGRAM_DIR,
    )
    assert proc.stdin is not None and proc.stdout is not None

    # Protocol sanity: the server must describe at least one ASR model.
    await async_write_event(Describe().event(), proc.stdin)
    info = Info.from_event(
        await _next_event_of(Info.is_type, proc.stdout, _INFO_TIMEOUT)
    )
    assert len(info.asr) == 1, "Expected one asr service"
    assert len(info.asr[0].models) > 0, "Expected at least one model"

    # Transcribe a known WAV.
    await async_write_event(Transcribe(language="en").event(), proc.stdin)
    with wave.open(str(_DIR / "turn_on_the_living_room_lamp.wav"), "rb") as wav:
        await async_write_event(
            AudioStart(
                rate=wav.getframerate(),
                width=wav.getsampwidth(),
                channels=wav.getnchannels(),
            ).event(),
            proc.stdin,
        )
        for chunk in wav_to_chunks(wav, _SAMPLES_PER_CHUNK):
            await async_write_event(chunk.event(), proc.stdin)
        await async_write_event(AudioStop().event(), proc.stdin)

    transcript = Transcript.from_event(
        await _next_event_of(Transcript.is_type, proc.stdout, _TRANSCRIBE_TIMEOUT)
    )
    text = re.sub(r"[^a-z ]", "", transcript.text.lower().strip())
    assert text == "turn on the living room lamp"

    proc.stdin.close()
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError:
        proc.terminate()
        _, stderr = await proc.communicate()

    if compute_type == "int8":
        # The int8 no-op warning added after benchmarking must be present.
        assert _INT8_WARNING_FRAGMENT in stderr.decode(), (
            "Expected the int8 reality-check warning in server logs"
        )
