# syntax=docker/dockerfile:1

# ===========================================================================
# Builder stage: full NGC TensorRT image (Python + build toolchain + the CUDA
# devel toolkit). Creates the venv and installs requirements + torch2trt. The
# heavyweight bits here — nvcc, CUDA headers/static libs (~3.8 GB), apt/pip
# build tooling — are needed only to build and are NOT carried into the
# runtime image.
# ===========================================================================
FROM nvcr.io/nvidia/tensorrt:26.07-py3 AS builder

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
# Make apt resilient to a stalled mirror connection: without a timeout apt can
# hang indefinitely on a half-open fetch (observed wedging a multi-hour build);
# retries + a 30s timeout make it fail fast and recover instead.
RUN printf 'Acquire::Retries "3";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' > /etc/apt/apt.conf.d/99network-resilience \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3-venv \
        git \
    && chmod +x ./script/setup \
    && ./script/setup \
    # Trim caches/build leftovers so the copied venv layer stays lean.
    && find /usr/src/wyoming-whisper-trt/.venv -type d -name __pycache__ -prune -exec rm -rf {} + \
    && rm -rf /root/.cache/pip /usr/src/wyoming-whisper-trt/.git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ===========================================================================
# Runtime stage: slim Ubuntu. This tag is NOT a free-floating dependency — it
# must stay on the same Ubuntu release as the NGC builder above.
# nvcr.io/nvidia/tensorrt:26.07-py3 is Ubuntu 24.04 (CUDA 13.3.1, TRT
# 11.1.0.106), so this is 24.04 and its python3.12. Ubuntu 26.04 was tried and
# reverted: it has no python3.12 package, and installing its python3.14
# instead left the venv interpreter unable to find its own stdlib
# ("ModuleNotFoundError: No module named 'encodings'").
# script/setup builds the venv with venv's POSIX default of
# symlinks, so the .venv copied in below points at the builder's interpreter
# path: a runtime without that exact python3.X gets a dangling symlink, and a
# different minor version can't load the cp3XX wheels in site-packages either.
# The verification step after the COPY below asserts this at build time rather
# than letting it surface as a broken container at run time. Bumping this in
# step with the NGC base is a deliberate, tested change.
#
# On top of the venv we need: the Python stdlib, ffmpeg (audio loading),
# libgomp (torch OpenMP), and
# ca-certificates (TLS trust store — the model is downloaded over HTTPS on
# first run; without it that fetch fails with CERTIFICATE_VERIFY_FAILED —
# unlike the NGC base, plain ubuntu ships no CA bundle). libcuda.so and
# nvidia-smi are injected by the NVIDIA container runtime. This drops the
# ~3.8 GB CUDA devel toolkit and the build toolchain from the image.
# ===========================================================================
FROM ubuntu:24.04 AS runtime

RUN printf 'Acquire::Retries "3";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' > /etc/apt/apt.conf.d/99network-resilience \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3.12 \
        ffmpeg \
        libgomp1 \
        ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Make the GPU visible when the caller doesn't set these (the NVIDIA runtime
# reads them to inject the driver, libcuda and nvidia-smi). "utility" provides
# nvidia-smi, which run.sh uses to key the engine cache by GPU arch.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Bring over the built application + self-contained venv.
COPY --from=builder /usr/src/wyoming-whisper-trt /usr/src/wyoming-whisper-trt

# Assert the relocated venv actually runs on this base. The venv's interpreter
# is a symlink back to the builder's python, and site-packages holds cp3XX
# wheels, so a runtime whose python minor version differs from the builder's
# produces an image that only fails when a container starts. Catch it here.
RUN set -eu; \
    # The app is not pip-installed into the venv (script/setup installs
    # requirements + torch2trt only), so wyoming_whisper_trt resolves via the
    # working directory, exactly as run.sh does before launching it.
    cd /usr/src/wyoming-whisper-trt; \
    py=/usr/src/wyoming-whisper-trt/.venv/bin/python3; \
    if ! "$py" -c 'import sys; print("venv python:", sys.version)'; then \
        echo "ERROR: the venv interpreter does not run on this runtime base." >&2; \
        echo "It resolves to: $(readlink -f "$py" 2>/dev/null || echo '<dangling>')" >&2; \
        echo "This runtime provides: $(ls -d /usr/lib/python3.* 2>/dev/null | tr '\n' ' ')" >&2; \
        echo "The runtime python minor version must match the NGC builder's." >&2; \
        exit 1; \
    fi; \
    "$py" -c 'import torch, wyoming_whisper_trt'

WORKDIR /
COPY ./run.sh ./

EXPOSE 10300

ENTRYPOINT ["bash", "/run.sh"]
