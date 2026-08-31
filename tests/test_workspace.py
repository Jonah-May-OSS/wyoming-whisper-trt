"""Unit tests for the automatic TensorRT workspace sizing.

These exercise ``auto_workspace_mb`` with ``torch.cuda.mem_get_info`` mocked,
so they run on CPU-only CI runners (no real GPU required).
"""

import warnings
from unittest.mock import patch

import pytest

from whisper_trt import auto_workspace_mb

# The documented contract values from whisper_trt.model (kept as literals here
# so the test pins the behavior rather than re-deriving it from the module).
_DEFAULT_WORKSPACE_MB = 1024
_LARGE_WORKSPACE_MB = 4096
_MIN_WORKSPACE_MB = 256
_VRAM_FRACTION = 0.5
_BUILD_MEMORY_RESERVE_MB = 2048

_MIB = 1 << 20


def _free_vram(mb: int):
    """Patch mem_get_info to report ``mb`` MiB free (total is irrelevant)."""
    return patch(
        "whisper_trt.model.torch.cuda.mem_get_info",
        return_value=(mb * _MIB, mb * _MIB),
    )


def test_large_model_gets_large_budget_when_vram_is_ample() -> None:
    # 16 GiB free: even after the reserve, 50% of the spare exceeds the 4 GiB
    # target, so no clamp.
    with _free_vram(16 * 1024):
        assert auto_workspace_mb("large-v3") == _LARGE_WORKSPACE_MB


def test_small_model_gets_default_budget_when_vram_is_ample() -> None:
    with _free_vram(16 * 1024):
        assert auto_workspace_mb("base") == _DEFAULT_WORKSPACE_MB


def test_budget_is_clamped_to_spare_vram_after_reserve() -> None:
    # 8 GiB free (e.g. an RTX 3050): after the 2 GiB build reserve, 6 GiB is
    # spare; the workspace takes 50% of that (~3 GiB), below the 4 GiB target.
    with _free_vram(8 * 1024):
        budget = auto_workspace_mb("large-v3")
    assert budget == int(_VRAM_FRACTION * (8 * 1024 - _BUILD_MEMORY_RESERVE_MB))
    assert budget < _LARGE_WORKSPACE_MB


def test_reserve_is_subtracted_before_the_fraction() -> None:
    # The reserve must come off the top: a naive "fraction of all free VRAM"
    # would give more than the reserve-aware cap for the same free VRAM.
    free_mb = 8 * 1024
    with _free_vram(free_mb):
        budget = auto_workspace_mb("large-v3")
    naive = int(_VRAM_FRACTION * free_mb)
    assert budget == int(_VRAM_FRACTION * (free_mb - _BUILD_MEMORY_RESERVE_MB))
    assert budget < naive


def test_budget_never_drops_below_the_floor() -> None:
    # Almost no free VRAM: spare goes negative after the reserve, so the floor
    # wins rather than a zero/negative workspace.
    with _free_vram(64):
        assert auto_workspace_mb("large-v3") == _MIN_WORKSPACE_MB


def test_floor_applies_when_free_equals_reserve() -> None:
    # Exactly the reserve free: zero spare, so the floor wins.
    with _free_vram(_BUILD_MEMORY_RESERVE_MB):
        assert auto_workspace_mb("large-v3") == _MIN_WORKSPACE_MB


@pytest.mark.parametrize("model", ["large", "large-v2", "large-v3", "large-v3-turbo"])
def test_all_large_variants_use_the_large_target(model: str) -> None:
    with _free_vram(16 * 1024):
        assert auto_workspace_mb(model) == _LARGE_WORKSPACE_MB


def test_falls_back_to_target_when_cuda_info_unavailable() -> None:
    # No CUDA device / not initialized: trust the model-size target, unclamped.
    with patch(
        "whisper_trt.model.torch.cuda.mem_get_info",
        side_effect=RuntimeError("no CUDA"),
    ):
        assert auto_workspace_mb("large-v3") == _LARGE_WORKSPACE_MB
        assert auto_workspace_mb("base") == _DEFAULT_WORKSPACE_MB


class TestEffectiveWorkspace:
    """The per-engine re-clamp applied just before each engine build.

    ``auto_workspace_mb`` runs once, before anything is resident. A "kv" build
    then makes four engines in sequence, each leaving weights in VRAM, so the
    free memory that sized the original budget is gone by the last build.
    ``_effective_workspace`` re-reads free VRAM before each one.
    """

    @pytest.fixture(autouse=True)
    def _restore_builder_state(self):
        # These set class attributes, so put them back or the next test in the
        # session inherits whatever the last one left behind.
        from whisper_trt.model import WhisperTRTBuilder

        size = WhisperTRTBuilder.max_workspace_size
        explicit = WhisperTRTBuilder.max_workspace_explicit
        yield
        WhisperTRTBuilder.max_workspace_size = size
        WhisperTRTBuilder.max_workspace_explicit = explicit

    def _builder(self, size_mb: int, explicit: bool):
        from whisper_trt.model import WhisperTRTBuilder

        WhisperTRTBuilder.max_workspace_size = size_mb * _MIB
        WhisperTRTBuilder.max_workspace_explicit = explicit
        return WhisperTRTBuilder

    def test_auto_budget_shrinks_as_vram_fills(self) -> None:
        builder = self._builder(4096, explicit=False)
        # Ample VRAM: the up-front budget survives untouched.
        with _free_vram(16 * 1024):
            assert builder._effective_workspace() == 4096 * _MIB
        # Three engines later there is far less free, so the stale ceiling is
        # cut down rather than letting tactic search reserve what isn't there.
        with _free_vram(4 * 1024):
            clamped = builder._effective_workspace()
        assert (
            clamped
            == int(_VRAM_FRACTION * (4 * 1024 - _BUILD_MEMORY_RESERVE_MB)) * _MIB
        )
        assert clamped < 4096 * _MIB

    def test_explicit_budget_is_never_shrunk(self) -> None:
        # --max-workspace-mb is a number the user asked for; silently reducing
        # it would make the flag a suggestion.
        builder = self._builder(4096, explicit=True)
        with _free_vram(3 * 1024):
            assert builder._effective_workspace() == 4096 * _MIB

    def test_never_drops_below_the_floor(self) -> None:
        builder = self._builder(4096, explicit=False)
        with _free_vram(_BUILD_MEMORY_RESERVE_MB):
            assert builder._effective_workspace() == _MIN_WORKSPACE_MB * _MIB

    def test_falls_back_to_the_budget_without_cuda_info(self) -> None:
        builder = self._builder(1024, explicit=False)
        with patch(
            "whisper_trt.model.torch.cuda.mem_get_info",
            side_effect=RuntimeError("no cuda"),
        ):
            assert builder._effective_workspace() == 1024 * _MIB


class TestBuildOomDetection:
    """Recognising a TensorRT build OOM from its misleading message."""

    @pytest.mark.parametrize(
        "message",
        [
            "Could not find any implementation for node {ForeignNode[...]}",
            "[graph] no implementation obeys reformatting-free rules",
            "Cuda Runtime (out of memory)",
            "std::bad_alloc: OutOfMemory",
        ],
    )
    def test_oom_shapes_are_recognised(self, message: str) -> None:
        from whisper_trt.model import _looks_like_build_oom

        assert _looks_like_build_oom(message)

    @pytest.mark.parametrize(
        "message",
        [
            "Unsupported ONNX opset version: 21",
            "Network has dynamic or shape inputs, but no optimization profile",
        ],
    )
    def test_unrelated_failures_are_left_alone(self, message: str) -> None:
        from whisper_trt.model import _looks_like_build_oom

        assert not _looks_like_build_oom(message)


class TestUnsupportedArchWarningFilter:
    """The Orin sm_87 startup warning is silenced, but only that one."""

    def _filtered(self, message: str) -> bool:
        """Return True when the installed filter would suppress ``message``."""
        from wyoming_whisper_trt.__main__ import _silence_unsupported_arch_warning

        with warnings.catch_warnings(record=True) as caught:
            warnings.resetwarnings()
            _silence_unsupported_arch_warning()
            warnings.warn(message, UserWarning, stacklevel=1)
            return not caught

    def test_orin_capability_warning_is_silenced(self) -> None:
        # The real message is multi-line; torch emits the "Found GPU" line
        # first and filters match from the start of the message.
        assert self._filtered(
            "Found GPU0 Orin which is of compute capability (CC) 8.7.\n"
            "The following list shows the CCs this version of PyTorch was "
            "built for and the hardware CCs it supports:\n- 9.0 which "
            "supports hardware CC range(90, 100)"
        )

    def test_other_warnings_still_surface(self) -> None:
        # A genuinely unsupported GPU, and unrelated warnings, must not be
        # swallowed by a filter aimed at one cosmetic message.
        assert not self._filtered(
            "NVIDIA GeForce GTX 780 with CUDA capability sm_35 is not "
            "compatible with the current PyTorch installation."
        )
        assert not self._filtered("some unrelated deprecation")
