# SPDX-License-Identifier: MIT

"""Availability check for the explicit INT8 (Q/DQ) encoder path.

TensorRT 11 removed implicit quantization: there is no calibrator to hand the
builder any more, and INT8 is no longer something the builder may pick where it
happens to be faster. Quantization has to be *in the graph* as
Quantize/Dequantize pairs, which also removes the old failure mode where an int8
build came out byte-for-byte identical to float16 because no INT8 tactic won on
timing.

The rewrite itself lives in torch2trt, which owns the ONNX export: after export
it runs NVIDIA ModelOpt's ONNX quantizer over the graph, calibrating activation
ranges on the mels ``model.py`` passes as ``int8_calib_dataset`` and casting
whatever stays unquantized (attention matmuls, norms, softmax) to FP16 in the
same pass. Doing both in one pass is required rather than incidental: ModelOpt's
AutoCast, which handles the FP16-only case, does not support graphs that already
carry Q/DQ.

This module only answers whether that path can run here, so the server can say
so at startup instead of failing deep inside a build.
"""

import importlib
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_INSTALL_HINT = (
    "INT8 needs explicit Q/DQ quantization of the ONNX graph, which is provided by "
    "NVIDIA ModelOpt ('nvidia-modelopt' in requirements.txt). Install it, or build "
    "with COMPUTE_TYPE=float16."
)


def int8_available() -> bool:
    """Whether an INT8 encoder can be built in this environment."""
    try:
        importlib.import_module("modelopt.onnx.quantization")
    except ImportError as exc:
        logger.debug("INT8 unavailable: %s. %s", exc, _INSTALL_HINT)
        return False
    return True
