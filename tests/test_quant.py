"""Unit tests for the explicit INT8 (Q/DQ) quantization of the audio encoder.

TensorRT 11 only honours INT8 that is already in the graph, so what matters is
that calibration produces a module whose ONNX export carries
QuantizeLinear/DequantizeLinear pairs. That is checkable on CPU — no GPU, no
TensorRT, no engine build — which is what these tests do.
"""

import collections

import onnx
import pytest
import torch
from torch import nn

from whisper_trt._quant import int8_available, quantize_encoder_int8

pytestmark = pytest.mark.skipif(
    not int8_available(), reason="nvidia-modelopt is not installed"
)

_N_MELS = 8
_N_STATE = 16
_N_FRAMES = 20


class _MiniEncoder(nn.Module):
    """The encoder's quantizable shapes (conv front end + projection) in miniature."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)  # deterministic weights, for the accuracy check
        self.conv = nn.Conv1d(_N_MELS, _N_STATE, 3, padding=1)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(_N_STATE, _N_STATE)
        self.ln = nn.LayerNorm(_N_STATE)

    def forward(
        self, x: torch.Tensor, positional_embedding: torch.Tensor
    ) -> torch.Tensor:
        x = self.gelu(self.conv(x))
        x = x.permute(0, 2, 1)
        x = x + positional_embedding
        return self.ln(self.proj(x))


def _calibration_set(n: int = 3, seed: int = 0) -> list[list[torch.Tensor]]:
    """Stand-in for the mel calibration set: the module's positional arguments.

    Seeded, because quantization error depends on the data the ranges were
    calibrated from and a drifting fixture makes the accuracy check flaky.
    """
    generator = torch.Generator().manual_seed(seed)
    pos = torch.randn(_N_FRAMES, _N_STATE, generator=generator)
    return [
        [torch.randn(1, _N_MELS, _N_FRAMES, generator=generator), pos] for _ in range(n)
    ]


def _export_op_counts(module: nn.Module, tmp_path) -> collections.Counter:
    """Export ``module`` through the same exporter torch2trt uses and count ops."""
    path = tmp_path / "model.onnx"
    inputs = _calibration_set(1)[0]
    torch.onnx.export(
        module,
        tuple(inputs),
        str(path),
        input_names=["x", "positional_embedding"],
        output_names=["output"],
        dynamic_axes={"x": {2: "frames"}},
        # torch2trt forces the TorchScript exporter on torch >= 2.9, and it is
        # the only one ModelOpt's quantizers emit Q/DQ through.
        dynamo=False,
    )
    return collections.Counter(n.op_type for n in onnx.load(str(path)).graph.node)


class TestQuantizeEncoderInt8:
    """Calibration turns the module into one that exports Q/DQ."""

    def test_export_carries_qdq_pairs(self, tmp_path) -> None:
        module = _MiniEncoder().eval()
        assert "QuantizeLinear" not in _export_op_counts(module, tmp_path)

        quantized = quantize_encoder_int8(module, _calibration_set())
        ops = _export_op_counts(quantized, tmp_path)
        # Both the conv and the projection contribute weight and activation
        # quantizers, and every Quantize is paired with a Dequantize.
        assert ops["QuantizeLinear"] >= 4
        assert ops["QuantizeLinear"] == ops["DequantizeLinear"]

    def test_quantizes_in_place(self) -> None:
        module = _MiniEncoder().eval()
        assert quantize_encoder_int8(module, _calibration_set()) is module

    def test_calibrated_output_stays_close(self) -> None:
        """Q/DQ costs accuracy, but INT8 over a calibrated range stays close.

        Compared against the output's own spread rather than an absolute
        tolerance: what matters is that quantization error is small relative to
        the signal, not that it is below some fixed epsilon.
        """
        module = _MiniEncoder().eval()
        calibration = _calibration_set()
        inputs = calibration[0]
        with torch.no_grad():
            reference = module(*inputs)
        quantize_encoder_int8(module, calibration)
        with torch.no_grad():
            quantized = module(*inputs)
        error = (quantized - reference).abs().mean().item()
        assert error < 0.1 * reference.std().item()

    def test_empty_calibration_set_is_rejected(self) -> None:
        """Without calibration data the quantizers have no ranges to export."""
        with pytest.raises(RuntimeError, match="calibration set is empty"):
            quantize_encoder_int8(_MiniEncoder().eval(), [])
