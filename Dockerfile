# Use the base TensorRT image
FROM nvcr.io/nvidia/tensorrt:26.06-py3

WORKDIR /usr/src/wyoming-whisper-trt

# Copy the checked-out source (including the torch2trt submodule) from the
# build context instead of cloning at build time. Cloning pulled the default
# branch at build time — unpinned and ignoring the exact ref being built; COPY
# makes the image reproducible and match the source under build.
COPY . /usr/src/wyoming-whisper-trt

# Install system deps and run setup (builds the venv + compiles torch2trt).
RUN apt-get update && apt-get install -y \
    python3-venv \
    ffmpeg \
    git \
    && chmod +x ./script/setup \
    && ./script/setup \
    && chmod +x ./script/run \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY ./run.sh ./
    
EXPOSE 10300
    
ENTRYPOINT ["bash", "/run.sh"]
