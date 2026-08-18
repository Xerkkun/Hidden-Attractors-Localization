"""Shared-library loading policy for optional native numerical kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def load_shared_library(path: Path, compiler: str) -> tuple[Any, str | None]:
    """Load a library, temporarily exposing its compiler runtime on Windows."""

    runtime_directory: str | None = None
    directory_handle = None
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        compiler_path = shutil.which(compiler)
        if compiler_path is not None:
            runtime_directory = str(Path(compiler_path).resolve().parent)
            directory_handle = os.add_dll_directory(runtime_directory)
    try:
        return ctypes.CDLL(str(path.resolve())), runtime_directory
    finally:
        if directory_handle is not None:
            directory_handle.close()
