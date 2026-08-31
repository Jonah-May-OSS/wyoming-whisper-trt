"""The mel must be synchronised onto the decode stream before the encoder runs.

``_audio_to_mel`` builds the mel on whichever stream the caller is using;
``_decode_mel`` then switches to ``self.stream`` and feeds it to the encoder.
Entering a stream context does *not* synchronise the two, so without an explicit
wait the encoder could read the mel while its producing kernels were still in
flight -- yielding a subtly wrong spectrogram and, from it, a confident silence
token. Measured on an RTX 4080: 126 of 2400 transcriptions of clean speech came
back as '.', '[ Silence ]' or '[no audio]'; with the wait, 0 of 2400. Both
decoders were affected, since both route through ``_decode_mel``.

The race itself is timing-dependent and cannot be reproduced deterministically,
so what is pinned here is the ordering contract that removes it: the decode
stream waits on the producing stream, and the mel is marked live on the decode
stream, both *before* the encoder is invoked. Ordering is the whole point -- a
wait issued after ``embed_audio`` would satisfy a naive "was it called?" check
while fixing nothing.

Runs without a GPU: the streams and the mel are stand-ins that record what was
called on them, in order.
"""

from contextlib import contextmanager
from typing import Any

import pytest

model_module = pytest.importorskip(
    "whisper_trt.model", reason="whisper_trt requires torch/tensorrt"
)


class _Recorder:
    """Shared, ordered log of the calls the contract cares about."""

    def __init__(self) -> None:
        self.events: list[str] = []


class _FakeStream:
    def __init__(self, name: str, rec: _Recorder) -> None:
        self.name = name
        self._rec = rec

    def wait_stream(self, other: "_FakeStream") -> None:
        self._rec.events.append(f"{self.name}.wait_stream({other.name})")


class _FakeMel:
    def __init__(self, rec: _Recorder) -> None:
        self._rec = rec

    def record_stream(self, stream: _FakeStream) -> None:
        self._rec.events.append(f"mel.record_stream({stream.name})")


class _FakeModel:
    """The minimum surface ``_decode_mel`` touches, all of it recording."""

    def __init__(self, rec: _Recorder) -> None:
        self._rec = rec
        self.stream = _FakeStream("decode", rec)
        self.dims = type("Dims", (), {"n_text_ctx": 8})()

    def embed_audio(self, mel: Any) -> str:
        self._rec.events.append("embed_audio")
        return "audio_features"

    def _get_tokenizer(self) -> str:
        return "tokenizer"

    def _configure_tokenizer(self, tokenizer: Any, language: str) -> None:
        return None

    def _prepare_prompt_tokens(self, *args: Any, **kwargs: Any) -> tuple:
        return ("out_tokens", 1, 1)

    def _decode_sequence(self, tokenizer: Any, request: Any) -> tuple:
        self._rec.events.append("decode_sequence")
        return ("text", [], 0.0)


@pytest.fixture(name="recorded")
def recorded_fixture(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Run ``_decode_mel`` against fakes and return the ordered call log."""
    rec = _Recorder()
    producer = _FakeStream("producer", rec)

    @contextmanager
    def fake_stream_ctx(stream: _FakeStream):
        rec.events.append(f"enter({stream.name})")
        yield
        rec.events.append(f"exit({stream.name})")

    monkeypatch.setattr(
        model_module.torch.cuda, "current_stream", lambda: producer
    )
    monkeypatch.setattr(model_module.torch.cuda, "stream", fake_stream_ctx)

    fake = _FakeModel(rec)
    model_module.WhisperTRT._decode_mel(
        fake,
        _FakeMel(rec),
        language="en",
        stream=False,
        initial_prompt=None,
    )
    return rec


def test_decode_stream_waits_on_the_producing_stream(recorded: _Recorder) -> None:
    """The mel's producer must be waited on, not merely switched away from."""
    assert "decode.wait_stream(producer)" in recorded.events


def test_mel_is_marked_live_on_the_decode_stream(recorded: _Recorder) -> None:
    """Without this the allocator may reissue the mel mid-encode."""
    assert "mel.record_stream(decode)" in recorded.events


def test_sync_happens_before_the_encoder_reads_the_mel(
    recorded: _Recorder,
) -> None:
    """The ordering *is* the fix; a late wait would be no fix at all."""
    events = recorded.events
    assert "embed_audio" in events, "encoder was never invoked"
    encoder = events.index("embed_audio")
    assert events.index("decode.wait_stream(producer)") < encoder
    assert events.index("mel.record_stream(decode)") < encoder


def test_wait_is_taken_on_the_stream_actually_in_use(
    recorded: _Recorder,
) -> None:
    """Guards the subtle trap: inside the context ``current_stream()`` is
    already the decode stream, so capturing the producer *after* entering
    would make the wait a self-wait and silently do nothing."""
    events = recorded.events
    assert events.index("decode.wait_stream(producer)") > events.index(
        "enter(decode)"
    )
    assert "decode.wait_stream(decode)" not in events
