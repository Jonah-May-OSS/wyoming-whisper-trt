# SPDX-License-Identifier: MIT

"""Explicit INT8 (Q/DQ) quantization of the audio encoder.

TensorRT 11 removed implicit quantization: there is no calibrator to hand the
builder any more, and INT8 is no longer something the builder may pick where it
happens to be faster. Quantization now has to be *in the graph* as
Quantize/Dequantize pairs, which also removes the old failure mode where an
int8 build came out byte-for-byte identical to float16 because no INT8 tactic
won on timing.

The flow is:

1. :func:`quantize_encoder_int8` swaps the encoder's ``Conv1d``/``Linear``
   layers for ModelOpt quantized equivalents and calibrates their activation
   ranges on real speech mels (per-channel INT8 weights, per-tensor INT8
   activations, ``max`` calibration).
2. torch2trt exports that module through the TorchScript ONNX exporter, where
   ModelOpt's quantizers emit ``QuantizeLinear``/``DequantizeLinear`` nodes.
3. TensorRT parses the Q/DQ graph and builds genuine INT8 kernels; everything
   left unquantized (attention matmuls, layer norms, softmax) rides the FP16
   cast that ``fp16_mode`` applies to the same graph.

Attention matmuls and norms are deliberately left alone — quantizing them needs
per-tensor ranges for intermediate activations that this calibration set does
not represent, and they are not where the encoder's FLOPs are.
"""

import copy
import logging
import time
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_INSTALL_HINT = (
    "INT8 needs explicit Q/DQ quantization, which is provided by NVIDIA ModelOpt "
    "('nvidia-modelopt' in requirements.txt). Install it, or build with "
    "COMPUTE_TYPE=float16."
)


def _load_mtq() -> Any:
    """Import ModelOpt's torch quantization API, or explain why int8 can't run."""
    try:
        import modelopt.torch.quantization as mtq
    except ImportError as exc:
        raise RuntimeError(f"{_INSTALL_HINT} (import failed: {exc})") from exc
    return mtq


def int8_available() -> bool:
    """Whether an INT8 encoder can be built in this environment."""
    try:
        _load_mtq()
    except RuntimeError:
        return False
    return True


def quantize_encoder_int8(
    module: nn.Module,
    calib_dataset: list[list[Any]],
    *,
    verbose: bool = False,
) -> nn.Module:
    """Insert and calibrate INT8 Q/DQ observers in ``module``.

    Args:
        module: The audio-encoder module about to be converted to TensorRT.
            Quantized in place; the same object is returned.
        calib_dataset: Calibration items, each the module's positional arguments
            (``[mel, positional_embedding]``). Every item is run once, so the
            observers see the real activation ranges of real speech.
        verbose: Log ModelOpt's per-layer quantization summary.

    Returns:
        nn.Module: The quantized module, ready for ONNX export with Q/DQ.

    Raises:
        RuntimeError: If ModelOpt is unavailable or calibration has nothing to
            calibrate on.
    """
    if not calib_dataset:
        raise RuntimeError(
            "INT8 calibration set is empty; cannot quantize the encoder."
        )

    mtq = _load_mtq()
    # INT8_DEFAULT_CFG is ModelOpt's CNN/vision recipe: per-channel (axis 0)
    # INT8 weights, per-tensor INT8 inputs, "max" calibration. Copied because
    # mtq.quantize() resolves the config in place.
    config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)

    def forward_loop(model: nn.Module) -> None:
        for item in calib_dataset:
            model(*item)

    logger.info(
        "Calibrating INT8 quantizers for the audio encoder on %d clips",
        len(calib_dataset),
    )
    started = time.monotonic()
    quantized = mtq.quantize(module, config, forward_loop=forward_loop)
    logger.info("INT8 calibration finished in %.1f s", time.monotonic() - started)
    if verbose:
        mtq.print_quant_summary(quantized)
    return quantized
