# syntax=docker/dockerfile:1

# ===========================================================================
# Builder stage: full NGC TensorRT image (Python + build toolchain + the CUDA
# devel toolkit). Creates the venv and installs requirements + torch2trt. The
# heavyweight bits here — nvcc, CUDA headers/static libs (~3.8 GB), apt/pip
# build tooling — are needed only to build and are NOT carried into the
# runtime image.
# ===========================================================================
FROM nvcr.io/nvidia/tensorrt:26.06-py3 AS builder

WORKDIR /usr/src/wyoming-whisper-trt

# Copy the checked-out source (including the torch2trt submodule) from the
# build context instead of cloning at build time. Cloning pulled the default
# branch at build time — unpinned and ignoring the exact ref being built; COPY
# makes the image reproducible and match the source under build.
COPY . /usr/src/wyoming-whisper-trt

# Build the venv (installs requirements + torch2trt). torch bundles its own
# CUDA/cuDNN wheels and tensorrt is a pip wheel, so the resulting venv is
# self-contained w.r.t. CUDA/TensorRT — nothing links back to the base's
# /usr/local/cuda toolkit at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-venv \
        git \
    && chmod +x ./script/setup \
    && ./script/setup \
    # Trim caches/build leftovers so the copied venv layer stays lean.
    && find /usr/src/wyoming-whisper-trt/.venv -type d -name __pycache__ -prune -exec rm -rf {} + \
    && rm -rf /root/.cache/pip /usr/src/wyoming-whisper-trt/.git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ===========================================================================
# Runtime stage: slim Ubuntu (matches the NGC base's Ubuntu 24.04 / glibc /
# python3.12, so the --copies venv relocates cleanly). Only the Python stdlib,
# ffmpeg (audio loading) and libgomp (torch OpenMP) are needed on top of the
# venv; libcuda.so and nvidia-smi are injected by the NVIDIA container runtime.
# This drops the ~3.8 GB CUDA devel toolkit and the build toolchain, which the
# self-contained venv makes redundant at runtime.
# ===========================================================================
FROM ubuntu:24.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3.12 \
        ffmpeg \
        libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Make the GPU visible when the caller doesn't set these (the NVIDIA runtime
# reads them to inject the driver, libcuda and nvidia-smi). "utility" provides
# nvidia-smi, which run.sh uses to key the engine cache by GPU arch.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Bring over the built application + self-contained venv.
COPY --from=builder /usr/src/wyoming-whisper-trt /usr/src/wyoming-whisper-trt

WORKDIR /
COPY ./run.sh ./

EXPOSE 10300

ENTRYPOINT ["bash", "/run.sh"]
