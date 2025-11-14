# Contributing to Wyoming Whisper TRT

Thank you for your interest in contributing to Wyoming Whisper TRT! This document provides guidelines and instructions for development.

## Development Setup

### Prerequisites

- Python 3.12 or higher
- NVIDIA GPU with CUDA support (or CPU for development)
- Git with submodules initialized

### Setting Up the Development Environment

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/Jonah-May-OSS/wyoming-whisper-trt.git
cd wyoming-whisper-trt
```

2. If you already cloned without submodules:
```bash
git submodule update --init --recursive
```

3. Run the setup script to create a virtual environment and install dependencies:
```bash
chmod +x script/setup
python script/setup --dev
```

This will install both production and development dependencies, including:
- pytest and pytest-asyncio for testing
- ruff for fast linting and formatting
- black and isort for code formatting
- mypy for type checking

## Code Quality Tools

This project uses modern Python development tools to maintain code quality:

### Ruff - Fast Linting and Formatting

[Ruff](https://github.com/astral-sh/ruff) is a fast, Rust-based linter and formatter that replaces multiple tools (flake8, isort, pylint, and more).

**Run linting checks:**
```bash
python script/lint
```

This runs:
- `ruff check` - Fast linting (checks for errors, style issues, etc.)
- `ruff format --check` - Verify code formatting
- `black --check` - Traditional formatting check
- `isort --check` - Import sorting check
- `mypy` - Type checking

**Auto-fix issues:**
```bash
python script/format
```

This runs:
- `ruff check --fix` - Auto-fix linting issues
- `ruff format` - Format code
- `black` - Additional formatting
- `isort` - Sort imports

### Pytest - Testing Framework

We use pytest with asyncio support for testing.

**Run tests:**
```bash
python script/test
```

**Run specific tests:**
```bash
source .venv/bin/activate
pytest tests/test_faster_whisper.py -v
pytest tests/test_faster_whisper.py::test_faster_whisper -v
```

**Test configuration** is in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Configuration Files

### pyproject.toml

The main configuration file for the project, containing:
- Project metadata and dependencies
- Pytest configuration
- Ruff linting and formatting rules
- Black, isort, and mypy settings

### setup.cfg (Legacy)

Contains legacy flake8 configuration. Most configuration has been migrated to `pyproject.toml`.

## Coding Standards

### Import Sorting

Imports should be sorted in the following order:
1. Standard library imports
2. Third-party library imports
3. Local application imports

Ruff and isort will handle this automatically.

### Type Hints

- Use type hints for function parameters and return values
- Use `from typing import` for complex types
- Prefer modern Python 3.12+ type syntax when possible

### Docstrings

- Use Google-style docstrings
- Include docstrings for public modules, classes, and functions
- Be descriptive but concise

### Line Length

- Maximum line length is 88 characters (Black/Ruff default)
- The formatter will handle this automatically

## Testing Guidelines

### Writing Tests

1. Place tests in the `tests/` directory
2. Name test files with `test_` prefix (e.g., `test_handler.py`)
3. Name test functions with `test_` prefix
4. Use pytest fixtures for common setup
5. Use `@pytest.mark.asyncio` for async tests

### Test Coverage

- Write tests for new features
- Update existing tests when modifying functionality
- Aim for meaningful test coverage, not just high percentages

## Pull Request Process

1. **Fork and create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Follow coding standards
   - Write tests for new functionality
   - Update documentation as needed

3. **Run quality checks:**
   ```bash
   python script/format  # Auto-format code
   python script/lint    # Check for issues
   python script/test    # Run tests
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Respond to review feedback**

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration:

- **Ruff**: Runs linting and format checks on all PRs
- **Tests**: Runs the full test suite (on self-hosted runners with GPU)

All checks must pass before a PR can be merged.

## Getting Help

- Open an issue for bugs or feature requests
- Check existing issues and pull requests first
- Provide detailed information and examples

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
