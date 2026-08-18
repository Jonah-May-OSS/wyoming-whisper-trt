"""Unit tests for the shared TensorRT engine scratch pool.

A Whisper checkpoint is up to four engines that never run concurrently, so they
can share one device-memory pool instead of reserving a private one each. The
part worth testing is the resize: engine sizes only become known as each engine
is deserialized, so adding a larger engine reallocates the buffer, and every
context already bound to the old address has to be re-bound. Missing that would
leave earlier engines executing against freed memory — silent corruption rather
than a crash.

Pools are allocated on CPU here so the logic is checkable without a GPU; the
device is the only thing that differs at runtime.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_T2T_DIR = Path(__file__).resolve().parent.parent / "torch2trt" / "torch2trt"


def _trt_module() -> ModuleType:
    """Import torch2trt's ``trt_module``, falling back to the checkout.

    Only importable as part of the ``torch2trt`` package (it uses relative
    imports), and the checkout's outer directory shadows that package name from
    the repo root — so the fallback puts the inner directory first on the path,
    exactly as ``test_quant`` does for the conversion module.
    """
    try:
        return importlib.import_module("torch2trt.trt_module")
    except ImportError:
        pass
    sys.path.insert(0, str(_T2T_DIR.parent))
    for name in [
        n for n in sys.modules if n == "torch2trt" or n.startswith("torch2trt.")
    ]:
        del sys.modules[name]
    return importlib.import_module("torch2trt.trt_module")


try:
    SharedDeviceMemory = _trt_module().SharedDeviceMemory
except (ImportError, AttributeError) as err:  # pragma: no cover - env dependent
    # AttributeError, not just ImportError: the loader is documented to fall back
    # to private pools on a torch2trt that predates SharedDeviceMemory, so that
    # combination has to skip here rather than error out during collection.
    pytest.skip(f"SharedDeviceMemory is unavailable: {err}", allow_module_level=True)


class _FakeContext:
    """Records every ``set_device_memory`` call, as TensorRT would receive it."""

    def __init__(self) -> None:
        self.bindings: list[tuple[int, int]] = []

    def set_device_memory(self, address: int, size: int) -> None:
        self.bindings.append((address, size))

    @property
    def address(self) -> int:
        assert self.bindings, "context was never given device memory"
        return self.bindings[-1][0]


class _FakeEngine:
    """An engine of a given scratch size that hands out ``_FakeContext``s."""

    def __init__(self, nbytes: int, v2: bool = True) -> None:
        if v2:
            self.device_memory_size_v2 = nbytes
        self.device_memory_size = nbytes
        self.context = _FakeContext()

    def create_execution_context_without_device_memory(self) -> _FakeContext:
        return self.context


@pytest.fixture
def pool() -> object:
    """A pool backed by host memory, so the logic runs without CUDA."""
    return SharedDeviceMemory(device="cpu")


class TestSizing:
    """The pool reserves the largest engine, not the sum of all of them."""

    def test_sizes_to_the_largest_engine(self, pool) -> None:
        for nbytes in (1 << 20, 4 << 20, 2 << 20):
            pool.add(_FakeEngine(nbytes))
        assert pool.nbytes == 4 << 20

    def test_starts_empty(self, pool) -> None:
        assert pool.nbytes == 0

    def test_prefers_the_v2_size(self, pool) -> None:
        engine = _FakeEngine(1 << 20)
        engine.device_memory_size = 8 << 20  # deprecated value must lose
        pool.add(engine)
        assert pool.nbytes == 1 << 20

    def test_falls_back_to_the_legacy_size(self, pool) -> None:
        pool.add(_FakeEngine(3 << 20, v2=False))
        assert pool.nbytes == 3 << 20

    def test_a_zero_scratch_engine_still_gets_an_address(self, pool) -> None:
        engine = _FakeEngine(0)
        pool.add(engine)
        assert engine.context.address != 0


class TestRebinding:
    """Every context ends up pointing at the pool's current buffer."""

    def test_growing_rebinds_the_earlier_contexts(self, pool) -> None:
        small = _FakeEngine(1 << 20)
        pool.add(small)
        first_address = small.context.address

        large = _FakeEngine(4 << 20)
        pool.add(large)

        assert small.context.address == large.context.address
        assert small.context.address != first_address
        assert small.context.bindings[-1][1] == pool.nbytes

    def test_shrinking_does_not_disturb_the_earlier_contexts(self, pool) -> None:
        large = _FakeEngine(4 << 20)
        pool.add(large)
        bindings_before = len(large.context.bindings)

        small = _FakeEngine(1 << 20)
        pool.add(small)

        assert len(large.context.bindings) == bindings_before
        assert small.context.address == large.context.address
        # A context that fits gets the whole buffer, not its own engine's size.
        assert small.context.bindings[-1][1] == 4 << 20

    def test_all_engines_share_one_address(self, pool) -> None:
        engines = [_FakeEngine(n << 20) for n in (1, 4, 2, 8, 3)]
        for engine in engines:
            pool.add(engine)
        addresses = {engine.context.address for engine in engines}
        sizes = {engine.context.bindings[-1][1] for engine in engines}
        assert len(addresses) == 1
        assert sizes == {8 << 20}


class TestFailures:
    """A context TensorRT refuses to create is an error, not a None binding."""

    def test_a_refused_context_raises(self, pool) -> None:
        engine = _FakeEngine(1 << 20)
        engine.create_execution_context_without_device_memory = lambda: None
        with pytest.raises(RuntimeError, match="without device memory"):
            pool.add(engine)
