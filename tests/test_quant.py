"""Unit tests for the explicit INT8 (Q/DQ) quantization of the audio encoder.

TensorRT 11 only honours INT8 that is already in the graph, so what matters is
that the rewritten graph carries Q/DQ pairs, keeps FP32 I/O, and is *valid* —
type inference included, since a graph whose QuantizeLinear scale no longer
matches its input type passes a shallow check and then fails to parse. All of
that is checkable on CPU: no GPU, no TensorRT, no engine build.

These exercise the same ModelOpt entry points torch2trt calls during conversion
(``torch2trt/precision.py``), because that is where the encoder's precision is
decided.
"""

import collections
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
import torch
from torch import nn

from whisper_trt._quant import int8_available

_T2T_DIR = Path(__file__).resolve().parent.parent / "torch2trt" / "torch2trt"


def _t2t(name: str) -> ModuleType:
    """Import a torch2trt module, falling back to the submodule checkout.

    ``script/setup`` installs torch2trt, but these tests need to run without a
    compiled install too (no CUDA toolkit on a CPU-only checkout), and both
    modules used here import nothing from torch2trt itself.
    """
    try:
        return importlib.import_module(f"torch2trt.{name}")
    except ImportError:
        pass
    key = f"_torch2trt_checkout_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, _T2T_DIR / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


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
        torch.manual_seed(seed)
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


def _calibration_set(n: int = 4, seed: int = 0) -> list[list[torch.Tensor]]:
    """Stand-in for the mel calibration set: the module's positional arguments."""
    generator = torch.Generator().manual_seed(seed)
    pos = torch.randn(_N_FRAMES, _N_STATE, generator=generator)
    return [
        [torch.randn(1, _N_MELS, _N_FRAMES, generator=generator), pos] for _ in range(n)
    ]


_INPUT_NAMES = ["x", "positional_embedding"]


def _export(module: nn.Module, path) -> str:
    """Export through the exporter and options torch2trt uses."""
    torch.onnx.export(
        module,
        tuple(_calibration_set(1)[0]),
        str(path),
        input_names=_INPUT_NAMES,
        output_names=["output"],
        dynamic_axes={"x": {2: "frames"}},
        dynamo=False,
    )
    return str(path)


def _ops(model: onnx.ModelProto) -> collections.Counter:
    return collections.Counter(node.op_type for node in model.graph.node)


def _elem_types(model: onnx.ModelProto) -> set[int]:
    return {
        vi.type.tensor_type.elem_type
        for vi in list(model.graph.input) + list(model.graph.output)
    }


def _quantize(src: str, dst: str, fp16: bool) -> onnx.ModelProto:
    """Run the ONNX INT8 rewrite exactly as torch2trt does."""
    precision = _t2t("precision")
    calibration = _calibration_set()
    flattener = _t2t("flattener").Flattener.from_value(calibration[0])
    arrays = precision.calibration_arrays(calibration, flattener, _INPUT_NAMES)
    # Each input is stacked the same number of times, which is how ModelOpt's
    # data reader recovers the iteration count.
    assert arrays["x"].shape[0] // 1 == len(calibration)
    assert arrays["positional_embedding"].shape[0] // _N_FRAMES == len(calibration)
    precision.quantize_onnx_int8(src, dst, arrays, fp16=fp16)
    return onnx.load(dst)


class TestOnnxInt8Quantization:
    """The rewritten graph is INT8, optionally FP16, and valid."""

    def test_inserts_qdq_pairs(self, tmp_path) -> None:
        """The rewrite is what puts INT8 in the graph; the export has none."""
        src = _export(_MiniEncoder().eval(), tmp_path / "model.onnx")
        assert "QuantizeLinear" not in _ops(onnx.load(src))

        ops = _ops(_quantize(src, str(tmp_path / "int8.onnx"), fp16=True))
        # The conv and the projection each contribute weight and activation
        # quantizers, and every Quantize is paired with a Dequantize.
        assert ops["QuantizeLinear"] >= 4
        assert ops["QuantizeLinear"] == ops["DequantizeLinear"]

    def test_int8_with_fp16_is_a_valid_graph(self, tmp_path) -> None:
        """The regression: casting a Q/DQ graph with AutoCast produced a
        QuantizeLinear whose scale type no longer matched its input, which only
        strict type inference catches."""
        src = _export(_MiniEncoder().eval(), tmp_path / "model.onnx")
        model = _quantize(src, str(tmp_path / "int8.onnx"), fp16=True)
        onnx.checker.check_model(model, full_check=True)

    def test_int8_without_fp16_is_a_valid_graph(self, tmp_path) -> None:
        """INT8 alone (float32 elsewhere) has to hold up the same way."""
        src = _export(_MiniEncoder().eval(), tmp_path / "model.onnx")
        model = _quantize(src, str(tmp_path / "int8.onnx"), fp16=False)
        onnx.checker.check_model(model, full_check=True)

    def test_graph_io_stays_fp32(self, tmp_path) -> None:
        """The runtime keeps feeding mels and reading features as FP32."""
        src = _export(_MiniEncoder().eval(), tmp_path / "model.onnx")
        model = _quantize(src, str(tmp_path / "int8.onnx"), fp16=True)
        assert _elem_types(model) == {onnx.TensorProto.FLOAT}

    def test_fp16_only_path_is_a_valid_graph(self, tmp_path) -> None:
        """float16 (the default compute type) goes through AutoCast instead."""
        src = _export(_MiniEncoder().eval(), tmp_path / "model.onnx")
        model = _t2t("precision").autocast_onnx_to_fp16(onnx.load(src))
        onnx.checker.check_model(model, full_check=True)
        assert _ops(model)["Cast"] > 0
        assert _elem_types(model) == {onnx.TensorProto.FLOAT}


class TestCalibrationArrays:
    """Calibration data is handed over in the layout ModelOpt expects."""

    def test_rejects_a_mismatched_item(self) -> None:
        """A calibration item that does not match the model's inputs is caught
        here rather than as an opaque failure inside the quantizer."""
        calibration = _calibration_set(1)
        flattener = _t2t("flattener").Flattener.from_value(calibration[0])
        with pytest.raises(ValueError, match="takes 3 inputs"):
            _t2t("precision").calibration_arrays(
                calibration, flattener, [*_INPUT_NAMES, "extra"]
            )

    def test_concatenates_along_the_batch_axis(self) -> None:
        """Every input is stacked the same number of times, which is how the
        data reader derives its iteration count."""
        calibration = _calibration_set(3)
        flattener = _t2t("flattener").Flattener.from_value(calibration[0])
        arrays = _t2t("precision").calibration_arrays(
            calibration, flattener, _INPUT_NAMES
        )
        assert arrays["x"].shape == (3, _N_MELS, _N_FRAMES)
        assert arrays["positional_embedding"].shape == (3 * _N_FRAMES, _N_STATE)
        assert arrays["x"].dtype == np.float32
