"""Unit tests for the start-up guards around dead TensorRT modules.

A ``TRTModule`` can come back from ``load_state_dict`` looking loaded while
being unusable: TensorRT's Python bindings return ``None`` instead of raising,
both from ``deserialize_cuda_engine`` and from ``create_execution_context``, and
torch2trt stores whichever ``None`` it got. The failure then surfaces only at
the first inference, as an ``AttributeError`` on ``NoneType`` deep inside
torch2trt.

Everything here mocks the TRT layer out, so these run on CPU-only CI runners.
"""

from typing import Any
from unittest.mock import patch

import pytest

from whisper_trt import EngineContextError, IncompatibleEngineError
from whisper_trt.model import _load_engine_module, load_trt_model


class _FakeTRTModule:
    """Stand-in for torch2trt's ``TRTModule`` with controllable load results."""

    def __init__(self, engine: Any, context: Any) -> None:
        self.engine = engine
        self.context = context

    def cuda(self) -> "_FakeTRTModule":
        return self

    def load_state_dict(self, state_dict: Any) -> None:
        """Accept the plan and keep whatever engine/context were configured."""


def _patch_module(engine: Any, context: Any):
    """Make ``_load_engine_module`` build a module with this engine/context."""
    return patch(
        "whisper_trt.model._new_trt_module",
        return_value=_FakeTRTModule(engine, context),
    )


def _checkpoint() -> dict[str, Any]:
    return {"prefill_engine": b"a serialized plan"}


@pytest.fixture(autouse=True)
def _no_gpu_side_effects():
    """Neutralize the helpers that would otherwise reach for a real device."""
    with (
        patch("whisper_trt.model._reclaim_memory"),
        patch("whisper_trt.model.get_device_arch_tag", return_value="sm89"),
    ):
        yield


def test_missing_engine_is_an_incompatible_engine_error() -> None:
    """A plan that will not deserialize should still ask for a rebuild."""
    with (
        _patch_module(engine=None, context=object()),
        pytest.raises(IncompatibleEngineError),
    ):
        _load_engine_module(_checkpoint(), "prefill_engine", "prefill")


def test_missing_context_is_an_engine_context_error() -> None:
    """A valid plan with no execution context is an OOM, not a bad cache."""
    with (
        _patch_module(engine=object(), context=None),
        pytest.raises(EngineContextError) as excinfo,
    ):
        _load_engine_module(_checkpoint(), "prefill_engine", "prefill")

    # Naming the engine is the whole point: three of them are loaded in a row
    # and the traceback used to point at none of them.
    assert "prefill" in str(excinfo.value)
    assert "out of" in str(excinfo.value).lower()


def test_missing_context_does_not_trigger_a_rebuild() -> None:
    """``load_trt_model`` rebuilds on IncompatibleEngineError and must not here.

    Rebuilding needs more VRAM than loading, so responding to an OOM by
    deleting a working multi-GB checkpoint and building it again would turn a
    recoverable failure into a destructive one.
    """
    assert not issubclass(EngineContextError, IncompatibleEngineError)


def test_live_module_is_returned() -> None:
    engine, context = object(), object()
    with _patch_module(engine=engine, context=context):
        module = _load_engine_module(_checkpoint(), "prefill_engine", "prefill")

    assert module.engine is engine
    assert module.context is context


def test_engine_state_is_popped_from_the_checkpoint() -> None:
    """The serialized plan must not stay referenced after deserialization."""
    checkpoint = _checkpoint()
    with _patch_module(engine=object(), context=object()):
        _load_engine_module(checkpoint, "prefill_engine", "prefill")

    assert "prefill_engine" not in checkpoint


class _FakeModel:
    """A loaded model whose warm-up transcription fails the way TRT does."""

    def __init__(self, error: BaseException | None = None) -> None:
        self._error = error
        self.transcribe_calls = 0

    def transcribe(self, *args: Any, **kwargs: Any) -> dict[str, str]:
        self.transcribe_calls += 1
        if self._error is not None:
            raise self._error
        return {"text": ""}


def _patch_builder(model: _FakeModel):
    """Point ``load_trt_model`` at a builder that returns ``model``."""

    class _FakeBuilder:
        @staticmethod
        def load(path: str) -> _FakeModel:
            return model

        @staticmethod
        def build(path: str, verbose: bool = False) -> None:  # pragma: no cover
            raise AssertionError("build must not be called; the checkpoint exists")

    return (
        patch.dict("whisper_trt.model.MODEL_BUILDERS", {"base.en": _FakeBuilder}),
        patch("whisper_trt.model.os.path.exists", return_value=True),
    )


def test_warm_up_failure_is_fatal() -> None:
    """A model that cannot transcribe silence must not be handed back.

    This is the regression that matters: the warm-up used to be wrapped in
    ``except (RuntimeError, ValueError)`` and logged at debug level, so a
    GPU-OOM start-up produced a server that answered every request with a
    failure while reporting itself healthy.
    """
    model = _FakeModel(
        AttributeError("'NoneType' object has no attribute 'set_tensor_address'")
    )
    builders, exists = _patch_builder(model)
    with builders, exists, pytest.raises(RuntimeError) as excinfo:
        load_trt_model("base.en", path="/nonexistent/base.en.pth", build=False)

    message = str(excinfo.value)
    assert "base.en" in message
    assert "set_tensor_address" in message
    assert model.transcribe_calls == 1


def test_warm_up_success_returns_the_model() -> None:
    model = _FakeModel()
    builders, exists = _patch_builder(model)
    with builders, exists:
        loaded = load_trt_model("base.en", path="/nonexistent/base.en.pth", build=False)

    assert loaded is model
    assert model.transcribe_calls == 1
