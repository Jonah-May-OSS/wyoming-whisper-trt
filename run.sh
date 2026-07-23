#!/bin/bash
set -euo pipefail

# Navigate to the application directory
cd /usr/src/wyoming-whisper-trt

# Check if the virtual environment is present; if not, run setup
if [ ! -d ".venv" ]; then
    echo "Virtual environment (.venv) not found. Running setup..."
    chmod +x script/setup
    setup_args=()
    if [ "${NVIDIA_EMBEDDED:-false}" = "true" ]; then
    echo "NVIDIA_EMBEDDED is true. Adjusting setup for embedded environment..."
    # Remove tensorrt and torch from requirements.txt if NVIDIA_EMBEDDED is true
    # This is necessary because the torch and tensorrt packages installed on NVIDIA embedded devices are not compatible 
    # with the versions specified in requirements.txt, and attempting to install them will cause conflicts and break the setup process.
        sed -i '/tensorrt/d;/torch/d' requirements.txt
    # Enable system site packages so VENV can access TensorRT and PyTorch installed on the system
        setup_args+=(--system_site_packages)
    fi
    ./script/setup "${setup_args[@]}"

fi

# Activate the Python virtual environment
source .venv/bin/activate

# Check if torch2trt is installed in this venv
python -c "import torch2trt" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "torch2trt not found. Installing via script/setup..."
    chmod +x script/setup
    ./script/setup
else
    echo "torch2trt is already installed. Skipping setup."
fi

# TRT engine plans are GPU-architecture specific. Suffix the engine dir with
# the GPU's compute capability so a plan built on one arch is never loaded on
# another (TRT deserialize "incompatible device" Error 6) if the container is
# rescheduled onto a different GPU. Model weights are arch-independent; only
# the engine cache (--data-dir) is keyed. Falls back to the bare dir when
# nvidia-smi is unavailable.
BASE_DATA_DIR="${DATA_DIR:-/data}"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -cd '0-9')"
if [ -n "$GPU_CC" ]; then
    ENGINE_DATA_DIR="${BASE_DATA_DIR}/sm${GPU_CC}"
else
    ENGINE_DATA_DIR="${BASE_DATA_DIR}"
fi
mkdir -p "$ENGINE_DATA_DIR"
echo "Using engine data dir: $ENGINE_DATA_DIR"

# Launch the main application
python3 -m wyoming_whisper_trt \
    --model "${MODEL:-base}" \
    --language "${LANGUAGE:-auto}" \
    --uri "${URI:-tcp://0.0.0.0:10300}" \
    --data-dir "$ENGINE_DATA_DIR" \
    --compute-type "${COMPUTE_TYPE:-float16}" \
    --decoder-mode "${DECODER_MODE:-kv}" \
    --device "${DEVICE:-cuda}" \
    --beam-size "${BEAM_SIZE:-5}" \
    --no-speech-threshold "${NO_SPEECH_THRESHOLD:-0.6}" \
    --silence-threshold "${SILENCE_THRESHOLD:-0.0}" \
    "$@"
