# Patches for torch2trt

This directory contains patches that are applied to the torch2trt submodule during setup to ensure compatibility with newer versions of dependencies.

## torch2trt-pytorch29-dynamic-shapes.patch

**Purpose**: Ensures compatibility with PyTorch 2.9+ ONNX export changes by forcing the legacy exporter path.

**Applied by**: The `script/setup` script automatically applies this patch before installing torch2trt.

**Details**: 
- PyTorch 2.9 changed the default ONNX export behavior to use the new `torch.export` path
- The new export path fails with torch2trt's `Flatten` module wrapper
- This patch forces `dynamo=False` to use the legacy ONNX exporter for PyTorch 2.9+
- The legacy exporter still uses `dynamic_axes` (not deprecated, just the default path changed)
- Maintains backward compatibility with PyTorch < 2.9

**Errors fixed**: 
```
# Failed to convert 'dynamic_axes' to 'dynamic_shapes'. 
Please provide 'dynamic_shapes' directly.
```
```
Failed to export the model with torch.export. 
[torch.onnx] Obtain model graph for `Flatten([...]` with `torch.export.export(..., strict=False)`... ❌
```
