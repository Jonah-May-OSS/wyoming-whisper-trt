"""Shared helpers for the developer tools in script/."""

import os
import subprocess
import sys
from pathlib import Path

PROGRAM_DIR = Path(__file__).resolve().parent.parent


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
    venv_python = PROGRAM_DIR / ".venv" / subdir
    if not venv_python.exists() or os.environ.get("_WWT_REEXECED") == "1":
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.environ["_WWT_REEXECED"] = "1"  # belt-and-braces against exec loops
    cmd = [str(venv_python), script_path, *sys.argv[1:]]
    if os.name == "nt":
        # Windows execv detaches from the console; run as a child instead.
        raise SystemExit(subprocess.call(cmd))
    os.execv(cmd[0], cmd)
