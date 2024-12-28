# Use the base TensorRT image
FROM nvcr.io/nvidia/tensorrt:24.12-py3

WORKDIR /usr/src

# Update and install python3-venv, clone the repository, and run the setup script
RUN apt-get update && apt-get install -y \
    python3-venv \
    ffmpeg \
    git \
    && git clone --recursive https://github.com/JonahMMay/wyoming-whisper-trt \
    && cd wyoming-whisper-trt \
    && chmod +x ./script/setup \
    && ./script/setup \
    && chmod +x ./script/run \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY ./run.sh ./
    
EXPOSE 10300
    
ENTRYPOINT ["bash", "/run.sh"]
