"""
Compatibility shim for onnx.helper functions removed in ONNX 1.20.0.

``onnx-graphsurgeon==0.5.8`` accesses ``onnx.helper.float32_to_bfloat16`` at
module-import time (in a module-level dict initializer).  That function was
deprecated in ONNX 1.18 and removed in ONNX 1.20.0.  On arm64 systems
(e.g. NVIDIA DGX Spark) pip resolves ONNX 1.20+, which causes an
``AttributeError`` before any model is loaded.

This module must be imported *before* ``torch2trt`` (and therefore before
``onnx_graphsurgeon``) so that the patched attributes are visible at the time
``onnx_graphsurgeon`` executes its module-level code.
"""

import math
import struct

import numpy as np
import onnx.helper

if not hasattr(onnx.helper, "float32_to_bfloat16"):

    def _float32_to_bfloat16(fval: float, truncate: bool = False) -> int:
        """Convert a float32 scalar to a bfloat16 value returned as int."""
        ival = int.from_bytes(struct.pack("<f", fval), "little")
        if truncate:
            return ival >> 16
        if math.isnan(fval):
            return 0x7FC0  # sign=0, exp=all-ones, sig=0b1000000
        rounded = ((ival >> 16) & 1) + 0x7FFF
        return (ival + rounded) >> 16

    onnx.helper.float32_to_bfloat16 = _float32_to_bfloat16  # type: ignore[attr-defined]

if not hasattr(onnx.helper, "float32_to_float8e4m3"):

    def _float32_to_float8e4m3(
        fval: float,
        scale: float = 1.0,
        fn: bool = True,
        uz: bool = False,
        saturate: bool = True,
    ) -> int:
        """Convert a float32 scalar to float8 e4m3 returned as int."""
        if not fn:
            raise NotImplementedError(
                "float32_to_float8e4m3 not implemented with fn=False."
            )
        x = fval / scale
        b = int.from_bytes(struct.pack("<f", np.float32(x)), "little")
        ret = (b & 0x80000000) >> 24  # sign
        if uz:
            if (b & 0x7FC00000) == 0x7FC00000:
                return 0x80
            if np.isinf(x):
                return ret | 127 if saturate else 0x80
            e = (b & 0x7F800000) >> 23
            m = b & 0x007FFFFF
            if e < 116:
                ret = 0
            elif e < 120:
                ex = e - 119
                if ex >= -2:
                    ret |= 1 << (2 + ex)
                    ret |= m >> (21 - ex)
                elif m > 0:
                    ret |= 1
                else:
                    ret = 0
                mask = 1 << (20 - ex)
                if m & mask and (
                    ret & 1
                    or m & (mask - 1) > 0
                    or (m & mask and m & (mask << 1) and m & (mask - 1) == 0)
                ):
                    ret += 1
            else:
                e8 = e - 119
                if e8 > 15 or (e8 == 15 and m > 0):
                    ret = ret | 127 if saturate else 0x80
                else:
                    ret |= e8 << 3
                    ret |= m >> 20
                    mask = 0x80000
                    if m & mask and (
                        ret & 1
                        or m & (mask - 1) > 0
                        or (m & mask and m & (mask << 1) and m & (mask - 1) == 0)
                    ):
                        ret += 1
            return ret
        # fn=True path
        if (b & 0x7FC00000) == 0x7FC00000:
            return ret | 0x7F
        if np.isinf(x):
            return (ret | 0x7E) if saturate else (ret | 0x7F)
        e = (b & 0x7F800000) >> 23
        m = b & 0x007FFFFF
        if e < 117:
            ret = 0
        elif e < 121:
            ex = e - 120
            if ex >= -2:
                ret |= 1 << (2 + ex)
                ret |= m >> (21 - ex)
            elif m > 0:
                ret |= 1
            else:
                ret = 0
            mask = 1 << (20 - ex)
            if m & mask and (
                ret & 1
                or m & (mask - 1) > 0
                or (m & mask and m & (mask << 1) and m & (mask - 1) == 0)
            ):
                ret += 1
        else:
            e8 = e - 120
            if e8 > 15 or (e8 == 15 and m > 0x400000):
                ret = (ret | 0x7E) if saturate else (ret | 0x7F)
            else:
                ret |= e8 << 3
                ret |= m >> 20
                mask = 0x80000
                if m & mask and (
                    ret & 1
                    or m & (mask - 1) > 0
                    or (m & mask and m & (mask << 1) and m & (mask - 1) == 0)
                ):
                    ret += 1
        return ret

    onnx.helper.float32_to_float8e4m3 = _float32_to_float8e4m3  # type: ignore[attr-defined]
