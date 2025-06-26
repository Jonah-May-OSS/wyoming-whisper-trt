#!/usr/bin/env bash
set -euo pipefail

# Navigate to the application directory
cd /usr/src/wyoming-whisper-trt

echo "▶️  Using Whisper model: $MODEL"

# Check if the virtual environment is present; if not, run setup
if [ ! -d ".venv" ]; then
    echo "Virtual environment (.venv) not found. Running setup..."
    chmod +x script/setup
    ./script/setup
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

# Launch the main application
exec python3 -m wyoming_whisper_trt.__main__ \
    --model "${MODEL:-base}" \
    --language "${LANGUAGE:-auto}" \
    --uri "${URI:-tcp://0.0.0.0:10300}" \
    --data-dir "${DATA_DIR:-/data}" \
    --download-dir "${DOWNLOAD_DIR:-/data}" \
    --device "${DEVICE:-cuda}" \
    ${DEBUG:+--debug}
