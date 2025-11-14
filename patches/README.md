# Patches for torch2trt

This directory contains patches that are applied to the torch2trt submodule during setup to ensure compatibility with newer versions of dependencies.

## torch2trt-pytorch29-dynamic-shapes.patch

**Purpose**: Adds support for PyTorch 2.9+ which deprecated `dynamic_axes` in favor of `dynamic_shapes` for ONNX export.

**Applied by**: The `script/setup` script automatically applies this patch before installing torch2trt.

**Details**: 
- PyTorch 2.9 introduced a breaking change where `dynamic_axes` parameter in `torch.onnx.export()` was deprecated
- The new API requires using `dynamic_shapes` with `torch.export.Dim` objects
- This patch adds version detection and uses the appropriate API based on the PyTorch version
- Maintains backward compatibility with PyTorch < 2.9

**Error fixed**: 
```
Failed to convert 'dynamic_axes' to 'dynamic_shapes'. 
Please provide 'dynamic_shapes' directly. 
Refer to the documentation for 'torch.export.export' for more information on dynamic shapes.
```
