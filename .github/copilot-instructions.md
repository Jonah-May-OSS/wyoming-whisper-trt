# Wyoming Whisper TRT - Copilot Instructions

This file provides guidance for GitHub Copilot coding agent when working on this repository. These instructions help ensure high-quality contributions that align with project standards and practices.

## Project Overview

Wyoming Whisper TRT is a Python-based speech recognition server that optimizes OpenAI Whisper with NVIDIA TensorRT for Home Assistant integration via the Wyoming Protocol. This provides significantly faster inference (~3x faster) while using less memory (~60% less) compared to standard PyTorch Whisper.

**Important**: Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Task Suitability and Delegation

### Suitable Tasks for Copilot Agent

This repository is well-suited for Copilot agent assistance on:
- **Bug fixes**: Fixing identified bugs in Python code, Docker configurations, or tests
- **Documentation updates**: Improving README, CONTRIBUTING, or inline documentation
- **Test coverage**: Adding or updating tests for existing functionality
- **Code refactoring**: Improving code structure while maintaining functionality
- **Feature additions**: Implementing well-defined, small to medium features
- **Dependency updates**: Updating package versions in requirements.txt
- **CI/CD improvements**: Enhancing GitHub Actions workflows

### Tasks Requiring Human Review

The following tasks require extra caution and human oversight:
- **Security-related changes**: Authentication, authorization, or secrets management
- **TensorRT model changes**: Core optimization logic that affects performance
- **CUDA/GPU integration**: Low-level GPU code that requires hardware testing
- **Breaking API changes**: Changes that affect Wyoming Protocol compatibility
- **Production configuration**: Changes to Docker images or deployment configs

### Task Expectations

When assigned a task:
1. Read the issue description carefully for context and acceptance criteria
2. Identify the minimal set of files that need modification
3. Make focused, surgical changes - avoid unnecessary refactoring
4. Follow existing code patterns and conventions
5. Test changes thoroughly before marking as complete
6. Update related documentation if API or behavior changes

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

## Coding Standards and Conventions

### Code Style and Formatting

This project uses modern Python development tools to maintain consistent code quality:

**Linters and Formatters:**
- **Ruff**: Fast, Rust-based linter and formatter (primary tool)
- **Black**: Code formatter with 88-character line length
- **isort**: Import sorting (standard library → third-party → local)
- **mypy**: Static type checking
- **flake8**: Additional style checking (legacy)

**Type Hints:**
- Use type hints for all function parameters and return values
- Use `from typing import` for complex types
- Prefer modern Python 3.12+ type syntax when possible

**Docstrings:**
- Use Google-style docstrings for public modules, classes, and functions
- Include descriptions, parameters, return values, and exceptions
- Be descriptive but concise

**Import Organization:**
```python
# Standard library imports
import os
import sys

# Third-party library imports
import numpy as np
import torch

# Local application imports
from wyoming_whisper_trt import handler
```

**Code Quality Requirements:**
- Maximum line length: 88 characters (enforced by Black/Ruff)
- All code must pass `python script/lint` before committing
- All code must be formatted with `python script/format`
- Write tests for new features (see `tests/` directory for examples)

### Pull Request and Contribution Workflow

When making changes to this repository:

1. **Before Starting:**
   - Read issue requirements carefully
   - Understand the scope and acceptance criteria
   - Identify files that need modification

2. **During Development:**
   - Make minimal, focused changes
   - Follow existing code patterns and conventions
   - Run `python script/format` to auto-format code
   - Run `python script/lint` frequently to catch issues early
   - Write or update tests for changed functionality

3. **Before Committing:**
   ```bash
   # Required pre-commit checks
   python script/format  # Auto-format (10-30 seconds)
   python script/lint    # Lint check (1-3 minutes, set 10+ min timeout)
   python script/test    # Run tests (10-15 minutes, set 30+ min timeout)
   ```

4. **Pull Request Guidelines:**
   - Write clear, descriptive commit messages
   - Reference related issues in the PR description
   - Explain what changed and why
   - Ensure all CI checks pass before requesting review
   - Respond to review feedback promptly

### Testing Guidelines

**Test Requirements:**
- Place tests in the `tests/` directory
- Name test files with `test_` prefix (e.g., `test_handler.py`)
- Name test functions with `test_` prefix
- Use pytest fixtures for common setup
- Use `@pytest.mark.asyncio` for async tests

**Running Tests:**
```bash
# Run all tests
python script/test

# Run specific test file
source .venv/bin/activate
pytest tests/test_faster_whisper.py -v

# Run specific test function
pytest tests/test_faster_whisper.py::test_faster_whisper -v
```

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

**⚠️ Partially Working Network Access:**
- **PyPI** (pypi.org): Intermittent access - Some packages install successfully, but experiencing timeout issues with larger installations

**Note**: Network connectivity issues may be related to known limitations of the GitHub Copilot coding agent environment. See [GitHub Community Discussion #171978](https://github.com/orgs/community/discussions/171978) for more context.

If you see `pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443)`, this indicates PyPI access is still experiencing reliability issues. The build process requires:
- Stable internet connection with high bandwidth  
- Access to multiple package repositories simultaneously
- No firewall restrictions on HTTPS traffic to all package indexes

**Partial Build Capability**: With current access, you can install:
- TensorRT packages (tensorrt-cu12-bindings, etc.)
- PyTorch with CUDA support
- Some PyPI packages (intermittently): Wyoming Protocol, development tools (black, isort, pytest)
- But NOT reliably: OpenAI Whisper, complex dependency chains

**Recommended Approach**: Continue using Docker builds or pre-built container images for reliable full builds. Individual package installations may work with retries.

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

## Copilot Agent Best Practices

### Working with This Repository

**Communication:**
- If stuck or uncertain, ask for clarification rather than making assumptions
- Explain your approach before making significant changes
- Document complex decisions in code comments or commit messages

**Iterative Development:**
- Make small, incremental changes rather than large refactors
- Test each change before moving to the next
- Use `report_progress` to share updates on long-running tasks

**Quality Assurance:**
- Always run formatters and linters before finalizing changes
- Ensure all tests pass, including new tests for new features
- Verify that changes don't break existing functionality
- Check that Docker configurations remain valid

**When in Doubt:**
- Refer to `CONTRIBUTING.md` for detailed development guidelines
- Check existing code for patterns and conventions
- Review test files for examples of testing approaches
- Consult the official documentation for dependencies (PyTorch, TensorRT, Wyoming)

### Common Issues and Solutions

**Network Timeout During Setup:**
- This is a known issue in restricted environments
- Solution: Use Docker builds or pre-built images
- See "Network Dependencies and Known Issues" section above

**CUDA/GPU Not Detected:**
- Verify NVIDIA drivers are installed
- Check NVIDIA Container Toolkit configuration
- Use `--device cpu` flag for development without GPU

**Test Failures:**
- Ensure all dependencies are installed (`python script/setup --dev`)
- Check that git submodules are initialized
- Verify model downloads completed successfully
- Review test logs for specific error messages

**Build Takes Too Long:**
- Expected: First build takes 45-90 minutes
- Don't cancel long-running operations
- Use appropriate timeouts as documented above
- Consider using pre-built Docker images for faster iteration