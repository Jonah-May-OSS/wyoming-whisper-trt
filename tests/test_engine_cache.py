"""Unit tests for hardware/TensorRT keying of the cached TRT engine.

TensorRT engine plans are only valid on the compute capability *and* the
TensorRT major version they were built on, so both are part of the cache
filename, and a plan that still fails to deserialize is discarded and rebuilt
once. These tests mock the CUDA capability query, the TensorRT version, and the
builder, so they run on CPU-only CI runners (no real GPU needed).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from whisper_trt import (
    IncompatibleEngineError,
    get_device_arch_tag,
    get_trt_version_tag,
)
from whisper_trt.model import get_model_filename, load_trt_model


def _capability(major: int, minor: int):
    """Patch the CUDA capability query to report ``major.minor``."""
    return patch(
        "whisper_trt.model.torch.cuda.get_device_capability",
        return_value=(major, minor),
    )


def _cuda_available(available: bool):
    return patch(
        "whisper_trt.model.torch.cuda.is_available",
        return_value=available,
    )


def _trt_version(version: str):
    """Patch the reported TensorRT version string."""
    return patch("whisper_trt.model.tensorrt.__version__", version)


def _tag_for(version: str) -> str:
    with _trt_version(version):
        return get_trt_version_tag()


class TestTrtVersionTag:
    """``get_trt_version_tag`` keys the cache on the exact TensorRT version."""

    def test_reports_the_full_version(self) -> None:
        with _trt_version("11.2.1.2"):
            assert get_trt_version_tag() == "trt11_2_1_2"

    def test_distinct_releases_never_share_a_tag(self) -> None:
        """Engines are not VERSION_COMPATIBLE, so even a patch bump needs its own
        plan: 11.2.1 cannot deserialize what 11.0.0 built."""
        tags = {
            _tag_for(version)
            for version in ("11.0.0.114", "11.1.0.106", "11.2.1.2", "10.16.1.11")
        }
        assert len(tags) == 4

    def test_suffixed_versions_stay_filename_safe(self) -> None:
        """Post/dev releases reach the filename with no dots or plus signs."""
        tag = _tag_for("10.7.0.post1")
        assert tag == "trt10_7_0_post1"
        assert tag.replace("_", "").isalnum()


class TestDeviceArchTag:
    """``get_device_arch_tag`` reports the device torch itself selects."""

    def test_reports_compute_capability(self) -> None:
        with _cuda_available(True), _capability(8, 6):
            assert get_device_arch_tag() == "sm86"

    def test_falls_back_when_cuda_is_unavailable(self) -> None:
        with _cuda_available(False):
            assert get_device_arch_tag() == "smunknown"

    def test_falls_back_when_the_driver_query_fails(self) -> None:
        with (
            _cuda_available(True),
            patch(
                "whisper_trt.model.torch.cuda.get_device_capability",
                side_effect=RuntimeError("no CUDA driver"),
            ),
        ):
            assert get_device_arch_tag() == "smunknown"


class TestArchKeyedFilename:
    """The cache filename separates plans built on different architectures."""

    def test_filename_carries_the_arch_and_trt_tags(self) -> None:
        with _cuda_available(True), _capability(8, 6), _trt_version("11.2.1.2"):
            assert (
                get_model_filename(
                    "base.en", "float16", decoder_mode="kv", max_workspace_mb=1024
                )
                == "base_en_trt_float16_kv4_ws1024_sm86_trt11_2_1_2.pth"
            )

    def test_different_trt_versions_never_share_a_filename(self) -> None:
        """A plan must never be handed to a TensorRT build that cannot load it."""
        args = ("base.en", "float16")
        kwargs = {"decoder_mode": "kv", "max_workspace_mb": 1024}
        with _cuda_available(True), _capability(8, 6):
            with _trt_version("11.0.0.114"):
                on_11_0 = get_model_filename(*args, **kwargs)
            with _trt_version("11.2.1.2"):
                on_11_2 = get_model_filename(*args, **kwargs)
            with _trt_version("10.16.1.11"):
                on_10 = get_model_filename(*args, **kwargs)
        assert len({on_11_0, on_11_2, on_10}) == 3

    def test_different_archs_never_share_a_filename(self) -> None:
        """The regression: an sm_86 plan must not be picked up on an sm_89 GPU."""
        args = ("base.en", "float16")
        kwargs = {"decoder_mode": "kv", "max_workspace_mb": 1024}
        with _cuda_available(True), _capability(8, 6):
            on_sm86 = get_model_filename(*args, **kwargs)
        with _cuda_available(True), _capability(8, 9):
            on_sm89 = get_model_filename(*args, **kwargs)
        assert on_sm86 != on_sm89


class _FakeModel:
    """Stand-in for a loaded WhisperTRT; only the warm-up call is exercised."""

    def transcribe(self, *_args, **_kwargs) -> dict[str, str]:
        return {"text": ""}


class _FakeBuilder:
    """Builder stub that fails to deserialize until the cache is rebuilt."""

    def __init__(self, path: Path, fail_loads: int) -> None:
        self._path = path
        self._remaining_failures = fail_loads
        self.builds = 0
        self.loads = 0

    def build(self, path: str, verbose: bool = False) -> None:
        self.builds += 1
        Path(path).write_bytes(b"engine")

    def load(self, path: str) -> _FakeModel:
        self.loads += 1
        if not Path(path).exists():
            raise FileNotFoundError(path)
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise IncompatibleEngineError("expecting compute 8.6 got compute 8.9")
        return _FakeModel()


@pytest.fixture(name="cached_engine")
def _cached_engine(tmp_path: Path) -> Path:
    """A pre-existing (stale) engine checkpoint in a shared cache dir."""
    path = tmp_path / "base_en_trt_float16_kv4_ws1024_sm86.pth"
    path.write_bytes(b"stale engine from another GPU")
    return path


def _load(path: Path, builder: _FakeBuilder, *, build: bool):
    """Invoke load_trt_model with the builder stubbed and warm-up disabled."""
    with patch.dict(
        "whisper_trt.model.MODEL_BUILDERS", {"base.en": builder}, clear=False
    ):
        return load_trt_model("base.en", path=str(path), build=build)


class TestIncompatibleEngineRecovery:
    """A plan that cannot deserialize here is rebuilt once, not crashed on."""

    def test_rebuilds_once_and_replaces_the_dead_cache(self, cached_engine) -> None:
        builder = _FakeBuilder(cached_engine, fail_loads=1)
        model = _load(cached_engine, builder, build=True)

        assert model is not None
        assert builder.builds == 1, "should rebuild exactly once"
        assert builder.loads == 2, "one failed load, then one after the rebuild"
        assert cached_engine.read_bytes() == b"engine", "stale plan was replaced"

    def test_propagates_when_building_is_disabled(self, cached_engine) -> None:
        builder = _FakeBuilder(cached_engine, fail_loads=1)
        with pytest.raises(IncompatibleEngineError):
            _load(cached_engine, builder, build=False)
        assert builder.builds == 0
        assert cached_engine.exists(), "cache must be left alone when build=False"

    def test_does_not_retry_forever(self, cached_engine) -> None:
        """A rebuild that still won't deserialize surfaces rather than looping."""
        builder = _FakeBuilder(cached_engine, fail_loads=2)
        with pytest.raises(IncompatibleEngineError):
            _load(cached_engine, builder, build=True)
        assert builder.builds == 1

    def test_tolerates_a_concurrent_process_unlinking_the_cache(
        self, cached_engine
    ) -> None:
        """Two processes sharing a cache dir can both discard the same file."""
        builder = _FakeBuilder(cached_engine, fail_loads=1)
        original_load = builder.load

        def load_then_race(path: str) -> object:
            try:
                return original_load(path)
            finally:
                Path(path).unlink(missing_ok=True)  # the other process got there first

        with patch.object(builder, "load", side_effect=load_then_race):
            model = _load(cached_engine, builder, build=True)
        assert model is not None
