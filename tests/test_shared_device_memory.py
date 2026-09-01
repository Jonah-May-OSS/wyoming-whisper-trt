"""Unit tests for the shared TensorRT engine scratch pool.

A Whisper checkpoint is up to four engines that never run concurrently, so they
can share one device-memory pool instead of reserving a private one each. Two
things here are worth testing without a GPU.

The resize: engine sizes only become known as each engine is deserialized, so
adding a larger engine reallocates the buffer, and every context already bound
to the old address has to be re-bound. Missing that would leave earlier engines
executing against freed memory -- silent corruption rather than a crash.

The API routes: asking TensorRT for a context with application-supplied memory
has been spelled three ways, and the oldest spelling was *removed* in TensorRT
11, which is what requirements.txt pins. Each route is exercised here, as is the
case where none exist -- that has to degrade to a private pool rather than fail
the load, since sharing is only an optimization.

Pools are allocated on CPU so the logic is checkable without a GPU; the device is
the only thing that differs at runtime.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_T2T_DIR = Path(__file__).resolve().parent.parent / "torch2trt" / "torch2trt"


def _trt_module() -> ModuleType:
    """Import torch2trt's trt_module, falling back to the checkout.

    Only importable as part of the ``torch2trt`` package (it uses relative
    imports), and the checkout's outer directory shadows that package name from
    the repo root -- so the fallback puts the inner directory first on the path,
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
    _MODULE = _trt_module()
    SharedDeviceMemory = _MODULE.SharedDeviceMemory
except (ImportError, AttributeError) as err:  # pragma: no cover - env dependent
    # AttributeError, not just ImportError: the loader is documented to fall back
    # to private pools on a torch2trt that predates SharedDeviceMemory, so that
    # combination has to skip here rather than error out during collection.
    pytest.skip(f"SharedDeviceMemory is unavailable: {err}", allow_module_level=True)


class _FakeContext:
    """Records every device-memory binding, as TensorRT would receive it."""

    def __init__(self, v2: bool = False) -> None:
        self.bindings: list[tuple[int, int]] = []
        if v2:
            self.set_device_memory_v2 = self._record
        else:
            self.set_device_memory = self._record

    def _record(self, address: int, size: int) -> None:
        self.bindings.append((address, size))

    @property
    def address(self) -> int:
        assert self.bindings, "context was never given device memory"
        return self.bindings[-1][0]


class _FakeRuntimeConfig:
    """The TensorRT 11 route: a config object carrying the strategy."""

    def __init__(self) -> None:
        self.strategy: Any = None

    def set_execution_context_allocation_strategy(self, strategy: Any) -> None:
        """Record the strategy the pool asked for."""
        self.strategy = strategy


class _FakeEngine:
    """An engine of a given scratch size, offering a chosen set of routes.

    ``routes`` names which context-creation spellings this TensorRT provides:
    "config" (11.x and 10.x), "strategy" (10.x), "legacy" (pre-10, removed in
    11.0), or none at all.
    """

    def __init__(
        self, nbytes: int, routes: tuple[str, ...] = ("config",), v2: bool = False
    ) -> None:
        self.device_memory_size_v2 = nbytes
        self.device_memory_size = nbytes
        self.context = _FakeContext(v2=v2)
        self.route_used: str | None = None
        self._routes = routes
        self.runtime_config: _FakeRuntimeConfig | None = None
        if "config" in routes:
            self.create_runtime_config = self._create_runtime_config
        if "legacy" in routes:
            self.create_execution_context_without_device_memory = self._legacy

    def _create_runtime_config(self) -> _FakeRuntimeConfig:
        self.runtime_config = _FakeRuntimeConfig()
        return self.runtime_config

    def _legacy(self) -> _FakeContext:
        self.route_used = "legacy"
        return self.context

    def create_execution_context(self, arg: Any = None) -> _FakeContext | None:
        """Accept whichever argument shape this fake claims to support."""
        if isinstance(arg, _FakeRuntimeConfig):
            self.route_used = "config"
            return self.context
        if arg is not None and "strategy" in self._routes:
            self.route_used = "strategy"
            return self.context
        raise TypeError("this engine does not accept that argument")


@pytest.fixture(name="pool")
def pool_fixture() -> Any:
    """A pool backed by host memory, so the logic runs without CUDA."""
    return SharedDeviceMemory(device="cpu")


@pytest.fixture(name="no_strategy")
def no_strategy_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the allocation-strategy enum, as a pre-10 TensorRT would."""
    monkeypatch.delattr(
        _MODULE.trt, "ExecutionContextAllocationStrategy", raising=False
    )


class TestRoutes:
    """Each spelling of a user-managed context request is tried in turn."""

    def test_prefers_the_runtime_config_route(self, pool: Any) -> None:
        engine = _FakeEngine(1 << 20, routes=("config", "strategy", "legacy"))
        assert pool.add(engine) is engine.context
        assert engine.route_used == "config"
        # The strategy has to reach the config, not merely be looked up.
        assert engine.runtime_config is not None
        assert engine.runtime_config.strategy is not None

    def test_falls_back_to_the_strategy_argument(self, pool: Any) -> None:
        engine = _FakeEngine(1 << 20, routes=("strategy", "legacy"))
        assert pool.add(engine) is engine.context
        assert engine.route_used == "strategy"

    @pytest.mark.usefixtures("no_strategy")
    def test_falls_back_to_the_legacy_entry_point(self, pool: Any) -> None:
        engine = _FakeEngine(1 << 20, routes=("legacy",))
        assert pool.add(engine) is engine.context
        assert engine.route_used == "legacy"

    @pytest.mark.usefixtures("no_strategy")
    def test_no_route_degrades_instead_of_raising(self, pool: Any) -> None:
        # TensorRT 11 removed the legacy entry point, so this is the shape of a
        # version offering nothing usable: it must not fail the load.
        engine = _FakeEngine(1 << 20, routes=())
        assert pool.add(engine) is None
        assert pool.nbytes == 0

    def test_a_refused_context_degrades(self, pool: Any) -> None:
        engine = _FakeEngine(1 << 20)
        engine.create_execution_context = lambda arg=None: None
        assert pool.add(engine) is None
        assert pool.nbytes == 0


class TestBinding:
    """Both device-memory setters are accepted, v2 preferred."""

    def test_uses_the_v2_setter_when_present(self, pool: Any) -> None:
        engine = _FakeEngine(2 << 20, v2=True)
        pool.add(engine)
        assert engine.context.bindings[-1][1] == 2 << 20

    def test_uses_the_legacy_setter_otherwise(self, pool: Any) -> None:
        engine = _FakeEngine(2 << 20, v2=False)
        pool.add(engine)
        assert engine.context.bindings[-1][1] == 2 << 20

    def test_an_unbindable_context_degrades(self, pool: Any) -> None:
        engine = _FakeEngine(1 << 20)
        del engine.context.set_device_memory
        assert pool.add(engine) is None


class TestSizing:
    """The pool reserves the largest engine, not the sum of all of them."""

    def test_sizes_to_the_largest_engine(self, pool: Any) -> None:
        for nbytes in (1 << 20, 4 << 20, 2 << 20):
            pool.add(_FakeEngine(nbytes))
        assert pool.nbytes == 4 << 20

    def test_starts_empty(self, pool: Any) -> None:
        assert pool.nbytes == 0

    def test_prefers_the_v2_size(self, pool: Any) -> None:
        engine = _FakeEngine(1 << 20)
        engine.device_memory_size = 8 << 20  # deprecated value must lose
        pool.add(engine)
        assert pool.nbytes == 1 << 20

    def test_falls_back_to_the_legacy_size(self, pool: Any) -> None:
        engine = _FakeEngine(3 << 20)
        del engine.device_memory_size_v2
        pool.add(engine)
        assert pool.nbytes == 3 << 20

    def test_a_zero_scratch_engine_still_gets_an_address(self, pool: Any) -> None:
        engine = _FakeEngine(0)
        pool.add(engine)
        assert engine.context.address != 0


class TestRebinding:
    """Every context ends up pointing at the pool's current buffer."""

    def test_growing_rebinds_the_earlier_contexts(self, pool: Any) -> None:
        small = _FakeEngine(1 << 20)
        pool.add(small)
        first_address = small.context.address

        large = _FakeEngine(4 << 20)
        pool.add(large)

        assert small.context.address == large.context.address
        assert small.context.address != first_address
        assert small.context.bindings[-1][1] == pool.nbytes

    def test_shrinking_does_not_disturb_the_earlier_contexts(self, pool: Any) -> None:
        large = _FakeEngine(4 << 20)
        pool.add(large)
        bindings_before = len(large.context.bindings)

        small = _FakeEngine(1 << 20)
        pool.add(small)

        assert len(large.context.bindings) == bindings_before
        assert small.context.address == large.context.address
        # A context that fits gets the whole buffer, not its own engine's size.
        assert small.context.bindings[-1][1] == 4 << 20

    def test_all_engines_share_one_address(self, pool: Any) -> None:
        engines = [_FakeEngine(n << 20) for n in (1, 4, 2, 8, 3)]
        for engine in engines:
            pool.add(engine)
        addresses = {engine.context.address for engine in engines}
        sizes = {engine.context.bindings[-1][1] for engine in engines}
        assert len(addresses) == 1
        assert sizes == {8 << 20}
