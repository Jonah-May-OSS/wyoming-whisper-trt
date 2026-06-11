"""Unit tests for the automatic TensorRT workspace sizing.

These exercise ``auto_workspace_mb`` with ``torch.cuda.mem_get_info`` mocked,
so they run on CPU-only CI runners (no real GPU required).
"""

from unittest.mock import patch

import pytest

from whisper_trt import auto_workspace_mb

# The documented contract values from whisper_trt.model (kept as literals here
# so the test pins the behavior rather than re-deriving it from the module).
_DEFAULT_WORKSPACE_MB = 1024
_LARGE_WORKSPACE_MB = 4096
_MIN_WORKSPACE_MB = 256
_VRAM_FRACTION = 0.7

_MIB = 1 << 20


def _free_vram(mb: int):
    """Patch mem_get_info to report ``mb`` MiB free (total is irrelevant)."""
    return patch(
        "whisper_trt.model.torch.cuda.mem_get_info",
        return_value=(mb * _MIB, mb * _MIB),
    )


def test_large_model_gets_large_budget_when_vram_is_ample() -> None:
    # 16 GiB free: 70% leaves far more than the 4 GiB target, so no clamp.
    with _free_vram(16 * 1024):
        assert auto_workspace_mb("large-v3") == _LARGE_WORKSPACE_MB


def test_small_model_gets_default_budget_when_vram_is_ample() -> None:
    with _free_vram(16 * 1024):
        assert auto_workspace_mb("base") == _DEFAULT_WORKSPACE_MB


def test_budget_is_clamped_to_a_fraction_of_free_vram() -> None:
    # 2 GiB free -> 70% = ~1434 MiB, below the 4 GiB large-model target.
    with _free_vram(2 * 1024):
        budget = auto_workspace_mb("large-v3")
    assert budget == int(_VRAM_FRACTION * 2 * 1024)
    assert budget < _LARGE_WORKSPACE_MB


def test_budget_never_drops_below_the_floor() -> None:
    # Almost no free VRAM: the 70% cap would be tiny, so the floor wins.
    with _free_vram(64):
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
