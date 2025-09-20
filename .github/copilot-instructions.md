# Wyoming Whisper TRT

Wyoming Whisper TRT is a Python-based speech recognition server that optimizes OpenAI Whisper with NVIDIA TensorRT for Home Assistant integration via the Wyoming Protocol. This provides significantly faster inference (~3x faster) while using less memory (~60% less) compared to standard PyTorch Whisper.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap, Build, and Setup Repository

**CRITICAL**: Always run these commands with LONG timeouts. Builds may take 45+ minutes. NEVER CANCEL long-running operations.

1. **Initialize Git Submodules (Required)**:
   ```bash
   git submodule update --init --recursive
   ```
   - Takes 1-2 minutes. Sets up torch2trt submodule dependency.

2. **Setup Development Environment**:
   ```bash
   chmod +x script/setup
   python script/setup --dev
   ```
   - **NEVER CANCEL**: Takes 45-60 minutes to complete. Set timeout to 90+ minutes.
   - Downloads and installs PyTorch, TensorRT, OpenAI Whisper, Wyoming Protocol, and development tools.
   - Creates `.venv` virtual environment.
   - Installs torch2trt from the git submodule.
   - **Network Requirements**: Requires internet access to PyPI, NVIDIA PyPI, and PyTorch repositories.
   - **Known Issue**: May fail with "Read timed out" errors in restricted network environments.

3. **Alternative Build (Without Dev Dependencies)**:
   ```bash
   python script/setup
   ```
   - Same timing and network requirements as above.

### Validation and Testing

**IMPORTANT**: All validation scripts require dependencies from `script/setup --dev` to be installed first.

1. **Validate Python Syntax (No Dependencies Required)**:
   ```bash
   python -m py_compile wyoming_whisper_trt/__init__.py
   python -m py_compile wyoming_whisper_trt/__main__.py
   python -m py_compile wyoming_whisper_trt/handler.py
   ```
   - Takes 5-10 seconds. Basic syntax validation without external dependencies.

2. **Format Code**:
   ```bash
   python script/format
   ```
   - Takes 10-30 seconds. Runs black and isort formatters.
   - **Requires**: Development dependencies installed via `script/setup --dev`

3. **Lint Code**:
   ```bash
   python script/lint
   ```
   - Takes 1-3 minutes. Runs black, isort, flake8, pylint, and mypy.
   - **NEVER CANCEL**: Set timeout to 10+ minutes for large codebases.
   - **Requires**: Development dependencies installed via `script/setup --dev`

4. **Validate Docker Configuration**:
   ```bash
   docker compose config
   ```
   - Takes 5-10 seconds. Validates docker-compose.yaml syntax and structure.

5. **Run Tests**:
   ```bash
   python script/test
   ```
   - **NEVER CANCEL**: Takes 10-15 minutes. Set timeout to 30+ minutes.
   - Tests Wyoming Protocol integration and speech recognition functionality.
   - Downloads tiny-int8 model if not present (requires ~200MB download).
   - **Requires**: Full dependencies installed via `script/setup --dev`

6. **Package for Distribution**:
   ```bash
   python script/package
   ```
   - Takes 1-2 minutes. Creates wheel distribution in `dist/` directory.

### Running the Application

1. **Local Development Run**:
   ```bash
   python script/run --help
   ```
   - Shows available command-line options.

2. **Basic Server Start**:
   ```bash
   python script/run --model base --uri tcp://127.0.0.1:10300 --data-dir ./data --download-dir ./download --device cuda
   ```
   - **NEVER CANCEL**: Initial run takes 30-45 minutes for model download and TensorRT optimization.
   - Set timeout to 60+ minutes for first run.

3. **Docker Development** (Recommended):
   ```bash
   docker compose up -d
   ```
   - **NEVER CANCEL**: Initial build takes 60-90 minutes. Set timeout to 120+ minutes.
   - Handles all dependencies and CUDA setup automatically.

## Quick Validation (No Dependencies Required)

For immediate code validation without waiting for full dependency installation:

```bash
# Validate Python syntax
python -m py_compile wyoming_whisper_trt/*.py

# Check git submodule status
git submodule status

# Validate Docker configuration
docker compose config

# Check repository structure
ls -la script/ wyoming_whisper_trt/ tests/

# Verify version information
cat wyoming_whisper_trt/VERSION
```

These commands help verify the codebase integrity before committing to long build processes.

## Manual Validation Requirements

### End-to-End Speech Recognition Testing

**ALWAYS test complete speech recognition workflows after making changes:**

1. **Start Server and Verify Listening**:
   ```bash
   # Terminal 1: Start server
   python -m wyoming_whisper_trt --model base --uri tcp://127.0.0.1:10300 --data-dir ./data --device cuda
   
   # Terminal 2: Test connectivity (in separate session)
   nc -z localhost 10300 && echo "Server is listening" || echo "Server not responding"
   ```

2. **Test with Sample Audio**:
   ```bash
   # Use the provided test audio file (RIFF WAVE, 16-bit, stereo 44100 Hz)
   python examples/transcribe.py base tests/turn_on_the_living_room_lamp.wav
   ```
   - Expected output should contain transcription of "turn on the living room lamp"
   - Test file: `tests/turn_on_the_living_room_lamp.wav` (527KB, stereo 44.1kHz)
   - **Requires**: Full dependencies and model downloads (30-45 minutes first run)

3. **Wyoming Protocol Integration Test**:
   ```bash
   python script/test
   ```
   - Validates complete Wyoming Protocol workflow including model loading, audio processing, and transcription.

## Build Timing and Expectations

- **Git submodule init**: 1-2 minutes
- **script/setup**: 45-60 minutes (NEVER CANCEL - set 90+ minute timeout)
- **script/lint**: 1-3 minutes (set 10+ minute timeout)
- **script/test**: 10-15 minutes (NEVER CANCEL - set 30+ minute timeout)
- **First application run**: 30-45 minutes for model download/optimization (set 60+ minute timeout)
- **Docker build**: 60-90 minutes (NEVER CANCEL - set 120+ minute timeout)

## Network Dependencies and Known Issues

**CRITICAL**: This project requires extensive network access:
- **PyPI** (pypi.org): For Python packages
- **NVIDIA PyPI** (pypi.nvidia.com): For TensorRT packages  
- **PyTorch Index** (download.pytorch.org): For PyTorch with CUDA
- **HuggingFace**: For Whisper model downloads

**Known Failure Mode**: In restricted network environments, pip install may fail with "Read timed out" errors. In such cases:
- Use Docker builds which have better network handling
- Use pre-built container images: `captnspdr/wyoming-whisper-trt:latest-amd64`
- Documentation note: Full build validation was limited in restricted environments due to PyPI connectivity issues

**Network Timeout Errors and Current Status**:

**✅ Working Network Access (as of latest testing):**
- **NVIDIA PyPI** (pypi.nvidia.com): Successfully accessible - TensorRT packages install correctly
- **PyTorch Index** (download.pytorch.org): Successfully accessible - PyTorch with CUDA installs correctly

**❌ Still Blocked Network Access:**
- **PyPI** (pypi.org): Still experiencing `ReadTimeoutError` - Wyoming Protocol, OpenAI Whisper, and development tools cannot be installed

If you see `pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443)`, this indicates PyPI access is still restricted. The build process requires:
- Stable internet connection with high bandwidth  
- Access to multiple package repositories simultaneously
- No firewall restrictions on HTTPS traffic to all package indexes

**Partial Build Capability**: With current access, you can install:
- TensorRT packages (tensorrt-cu12-bindings, etc.)
- PyTorch with CUDA support
- But NOT: Wyoming Protocol, OpenAI Whisper, development tools (black, isort, pytest)

**Recommended Approach**: Continue using Docker builds or pre-built container images until full PyPI access is available.

## GPU and CUDA Requirements

- **Required**: NVIDIA GPU with CUDA compute capability 7.0+
- **Required**: NVIDIA Container Toolkit (for Docker)
- **Required**: CUDA 12.8+ and TensorRT 10.13+
- **Development**: Can fall back to CPU mode with `--device cpu` but significantly slower

## CI/CD Integration

**Prerequisite**: Always run `script/setup --dev` first to install development dependencies.

Always run before committing:
```bash
# Quick syntax check (no dependencies required)
python -m py_compile wyoming_whisper_trt/*.py

# Full validation (requires dev dependencies)
python script/format  # Auto-format code (10-30 seconds)
python script/lint     # Verify code quality - NEVER CANCEL, set 10+ minute timeout
python script/test     # Run full test suite - NEVER CANCEL, set 30+ minute timeout

# Docker validation
docker compose config  # Validate Docker configuration (5-10 seconds)
```

## Key Files and Directories

### Repository Structure
```
.
├── script/                    # Build and development scripts
│   ├── setup                 # Environment setup (45-60 minutes)
│   ├── test                  # Test runner (10-15 minutes) 
│   ├── lint                  # Code linting (1-3 minutes)
│   ├── format                # Code formatting (10-30 seconds)
│   ├── run                   # Application runner
│   └── package               # Distribution packaging
├── wyoming_whisper_trt/      # Main Python package
├── whisper_trt/              # Whisper TensorRT optimizations
├── torch2trt/                # Git submodule for TensorRT conversions
├── tests/                    # Test suite and sample audio
├── examples/                 # Usage examples and benchmarking
└── requirements.txt          # Python dependencies
```

### Important Configuration Files
- `setup.py` - Package configuration and console script entry point
- `requirements.txt` - Production dependencies (PyTorch, TensorRT, Wyoming)
- `requirements_dev.txt` - Development dependencies (black, pytest, etc.)
- `setup.cfg` - Linting configuration (flake8, isort)

## Common Development Tasks

### Adding New Whisper Models
1. Update `wyoming_whisper_trt/handler.py` model validation
2. Test with `python script/test`
3. Validate Docker builds work with new models

### Debugging Speech Recognition Issues
1. Enable debug logging: `--debug` flag
2. Test with sample audio: `tests/turn_on_the_living_room_lamp.wav`
3. Check TensorRT model optimization logs
4. Verify GPU memory usage and CUDA compatibility

### Performance Optimization
1. Use `examples/profile_backend.py` for benchmarking
2. Compare against PyTorch Whisper and Faster-Whisper
3. Monitor GPU memory consumption during inference
4. Test different compute types (float16 vs float32)

## Docker Usage (Recommended for Production)

The project is designed to run in Docker containers with NVIDIA GPU support:

```yaml
# docker-compose.yaml example
services:
  wyoming-whisper-trt:
    image: captnspdr/wyoming-whisper-trt:latest-amd64
    environment:
      MODEL: "base"
      LANGUAGE: "auto" 
      DEVICE: "cuda"
      COMPUTE_TYPE: "float16"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Always use the provided Docker images for production deployments as they handle all CUDA and TensorRT dependencies correctly.