"""Shared helpers for the developer tools in script/."""

import os
import subprocess
import sys
import venv
from pathlib import Path

PROGRAM_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = PROGRAM_DIR / ".venv"
MODULE_DIR = PROGRAM_DIR / "wyoming_whisper_trt"
TESTS_DIR = PROGRAM_DIR / "tests"


def venv_python() -> str:
    """Return the repo .venv interpreter path, ensuring its dirs exist."""
    return venv.EnvBuilder().ensure_directories(VENV_DIR).env_exe


def ensure_venv_python(script_path: str) -> None:
    """Re-exec under the repo's .venv python if not already running in it.

    The developer tools need torch/tensorrt from .venv, but are commonly
    invoked with the system interpreter (``./script/benchmark``). Mirrors
    what script/run does rather than failing with ModuleNotFoundError.

    Args:
        script_path: The path to the script being executed, used to construct
            the re-exec command with the venv interpreter.

    Returns:
        None

    Raises:
        SystemExit: When running on Windows and re-execution is needed (raised
            from subprocess.call to propagate the child's exit code).
    """
    subdir = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    venv_python_path = PROGRAM_DIR / ".venv" / subdir
    if not venv_python_path.exists() or os.environ.get("_WWT_REEXECED") == "1":
        return
    if Path(sys.executable).resolve() == venv_python_path.resolve():
        return
    os.environ["_WWT_REEXECED"] = "1"  # belt-and-braces against exec loops
    cmd = [str(venv_python_path), script_path, *sys.argv[1:]]
    if os.name == "nt":
        # Windows execv detaches from the console; run as a child instead.
        raise SystemExit(subprocess.call(cmd))
    os.execv(cmd[0], cmd)
