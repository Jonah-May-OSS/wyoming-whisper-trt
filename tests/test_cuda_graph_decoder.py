"""The control flow around the CUDA-graph decode step.

No GPU: what is covered here is which engines a setting selects, that the two
step engines cannot be loaded into each other's decoder, and that step()
behaves the same whether or not capture succeeded. The arithmetic itself is
the engines' and is covered by the end-to-end test on the GPU runner.
"""

import pytest
import torch
from torch import nn

from whisper_trt._decoder import TextDecoderTRTKVGraph
from whisper_trt.model import WhisperTRTBuilder, get_model_filename


@pytest.fixture(autouse=True)
def _restore_builder_settings():
    """The builder's mode is class state, so put it back after each test."""
    mode, graphs = WhisperTRTBuilder.decoder_mode, WhisperTRTBuilder.cuda_graphs
    yield
    WhisperTRTBuilder.decoder_mode = mode
    WhisperTRTBuilder.cuda_graphs = graphs


class TestEffectiveDecoderMode:
    """Which step engine a user-facing setting resolves to."""

    @pytest.mark.parametrize(
        ("decoder_mode", "cuda_graphs", "expected"),
        [
            ("kv", True, "kv_graph"),
            ("kv", False, "kv"),
            # cuda_graphs only picks between kv step engines; the single-engine
            # decoder has no per-token step to capture.
            ("simple", True, "simple"),
            ("simple", False, "simple"),
        ],
    )
    def test_resolves(
        self, decoder_mode: str, cuda_graphs: bool, expected: str
    ) -> None:
        WhisperTRTBuilder.decoder_mode = decoder_mode
        WhisperTRTBuilder.cuda_graphs = cuda_graphs
        assert WhisperTRTBuilder.effective_decoder_mode() == expected


def test_the_two_step_engines_get_different_checkpoints() -> None:
    """Otherwise a graph engine loads into the dynamic decoder, or the reverse.

    The engines differ in shape -- one is fixed-capacity -- so the failure
    would not be a clean error, and both builds are keyed on the same model
    name and compute type. The filename is what keeps them apart.
    """
    kv = get_model_filename("base.en", "float16", decoder_mode="kv")
    graph = get_model_filename("base.en", "float16", decoder_mode="kv_graph")
    simple = get_model_filename("base.en", "float16", decoder_mode="simple")
    assert len({kv, graph, simple}) == 3


class _StubGraphDecoder(TextDecoderTRTKVGraph):
    """The graph decoder with its device-side setup left out.

    Everything step() branches on is plain Python state, so the fallback and
    the capacity guard can be exercised without TensorRT or a device.
    """

    def __init__(self, capacity: int = 4, graph: object | None = None) -> None:
        nn.Module.__init__(self)
        self.capacity = capacity
        self._graph = graph
        self._ready = True
        self._host_pos = 0
        self._token = torch.tensor([7])
        self.body_calls = 0

    def _step_body(self) -> None:
        self.body_calls += 1


class _FakeGraph:
    """Stands in for a captured torch.cuda.CUDAGraph."""

    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


class TestStep:
    """step() must behave identically whether or not capture succeeded."""

    def test_replays_the_graph_when_there_is_one(self) -> None:
        graph = _FakeGraph()
        decoder = _StubGraphDecoder(graph=graph)
        assert decoder.step() == 7
        assert graph.replays == 1
        assert decoder.body_calls == 0

    def test_runs_the_body_when_capture_failed(self) -> None:
        """The documented degradation: correct decoding, no launch saving.

        A driver or TensorRT combination that refuses to capture must not break
        transcription, so this path has to stay equivalent.
        """
        decoder = _StubGraphDecoder(graph=None)
        assert decoder.step() == 7
        assert decoder.body_calls == 1

    def test_both_paths_advance_the_position(self) -> None:
        replayed = _StubGraphDecoder(graph=_FakeGraph())
        direct = _StubGraphDecoder(graph=None)
        for _ in range(3):
            replayed.step()
            direct.step()
        assert replayed._host_pos == direct._host_pos == 3

    def test_stepping_before_begin_is_an_error(self) -> None:
        decoder = _StubGraphDecoder()
        decoder._ready = False
        with pytest.raises(RuntimeError, match="begin"):
            decoder.step()


class TestCapacity:
    """The cache is fixed-size, so the loop has to be told when it is full."""

    def test_room_up_to_capacity(self) -> None:
        decoder = _StubGraphDecoder(capacity=3, graph=_FakeGraph())
        assert decoder.can_step()
        for _ in range(3):
            assert decoder.can_step()
            decoder.step()
        assert not decoder.can_step()

    def test_a_zero_capacity_decoder_never_steps(self) -> None:
        assert not _StubGraphDecoder(capacity=0).can_step()
