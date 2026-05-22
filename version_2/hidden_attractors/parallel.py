#!/usr/bin/env python3
"""Shared parallelism policy for the fractional Chua backends.

Stability: internal
    Compilation helpers, OpenMP flags, and process-pool policy.  These are
    consumed by workflows and backends.  May change as new C kernels or
    platforms are added.

The numerical stages in this project mix causal fractional integrations,
OpenMP-parallel C kernels, Python process pools, and external backend
executables.  This module keeps the mechanical policy in one place:

- OpenMP compilation flags are platform-specific.
- Shared libraries for ctypes are built with ``-shared`` and ``-fPIC`` on
  POSIX systems.
- Standalone executables are never built with ``-shared`` or ``-fPIC``.
- OpenMP fallback is explicit and requires ``ALLOW_NO_OPENMP=1``.
- Python process workers force OpenMP to one thread inside each worker.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, MutableMapping, Sequence


ALLOW_NO_OPENMP_ENV = "ALLOW_NO_OPENMP"


@dataclass(frozen=True)
class CompileResult:
    path: Path
    command: List[str]
    openmp_requested: bool
    openmp_active: bool
    compiler: str
    target_kind: str


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "si", "s"}


def allow_no_openmp() -> bool:
    return env_flag(ALLOW_NO_OPENMP_ENV, False)


def force_single_openmp_thread_env(env: MutableMapping[str, str] | None = None) -> MutableMapping[str, str]:
    """Return an environment where any nested OpenMP runtime is single-threaded."""
    target = os.environ.copy() if env is None else env
    target["OMP_NUM_THREADS"] = "1"
    target["OMP_THREAD_LIMIT"] = "1"
    return target


def force_single_openmp_thread_current_process() -> None:
    """Apply the worker-side rule for Python multiprocessing tasks."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "1"


def distribute_openmp_threads(total_threads: int, external_processes: int) -> int:
    """Threads per external process when several independent processes are launched."""
    total = max(1, int(total_threads))
    processes = max(1, int(external_processes))
    return max(1, total // processes)


def _brew_libomp_prefix() -> Path:
    def validate(prefix: Path) -> Path:
        resolved = prefix.resolve()
        header = resolved / "include" / "omp.h"
        dylib = resolved / "lib" / "libomp.dylib"
        archive = resolved / "lib" / "libomp.a"
        if not header.exists() or (not dylib.exists() and not archive.exists()):
            raise RuntimeError(
                "La ruta de libomp es "
                f"{resolved}, pero no se encontro omp.h y libomp en include/lib. "
                "Reinstala con `brew install libomp` o define LIBOMP_PREFIX a una "
                "instalacion valida; para compilar sin OpenMP de forma explicita usa "
                f"{ALLOW_NO_OPENMP_ENV}=1."
            )
        return resolved

    raw = os.environ.get("LIBOMP_PREFIX")
    if raw:
        return validate(Path(raw).expanduser())
    brew = shutil.which("brew")
    if not brew:
        raise RuntimeError(
            "OpenMP en macOS requiere libomp de Homebrew. Instala libomp o define "
            "LIBOMP_PREFIX; para compilar sin OpenMP de forma explicita usa "
            f"{ALLOW_NO_OPENMP_ENV}=1."
        )
    env = os.environ.copy()
    env["HOMEBREW_NO_AUTO_UPDATE"] = "1"
    proc = subprocess.run(
        [brew, "--prefix", "libomp"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    prefix = proc.stdout.strip()
    if not prefix:
        raise RuntimeError("brew --prefix libomp no devolvio una ruta valida.")
    return validate(Path(prefix))


def _compiler_and_flags(openmp: bool, target_kind: str) -> tuple[str, List[str], List[str]]:
    system = platform.system().lower()
    if target_kind not in {"shared", "executable"}:
        raise ValueError("target_kind debe ser 'shared' o 'executable'.")

    if system == "darwin":
        compiler = "clang"
        cflags: List[str] = []
        ldflags: List[str] = []
        if openmp:
            prefix = _brew_libomp_prefix()
            cflags.extend(["-Xpreprocessor", "-fopenmp", f"-I{prefix / 'include'}"])
            ldflags.extend([f"-L{prefix / 'lib'}", "-lomp"])
        return compiler, cflags, ldflags

    if system == "windows":
        compiler = os.environ.get("CC", "gcc")
        flag = ["-fopenmp"] if openmp else []
        return compiler, flag, []

    compiler = os.environ.get("CC", "gcc")
    flag = ["-fopenmp"] if openmp else []
    return compiler, flag, []


def build_c_compile_command(source: Path, output: Path, *, target_kind: str, openmp: bool) -> List[str]:
    compiler, cflags, ldflags = _compiler_and_flags(openmp, target_kind)
    cmd = [compiler, "-O3"]
    if target_kind == "shared":
        cmd.append("-shared")
        if platform.system().lower() != "windows":
            cmd.append("-fPIC")
    cmd.extend(cflags)
    cmd.extend(["-o", str(output), str(source), "-lm"])
    cmd.extend(ldflags)
    return cmd


def _format_compile_failure(cmd: Sequence[str], exc: subprocess.CalledProcessError) -> str:
    stdout = (exc.stdout or "").strip()
    stderr = (exc.stderr or "").strip()
    parts = [
        "Fallo la compilacion C.",
        "Comando: " + " ".join(str(x) for x in cmd),
    ]
    if stdout:
        parts.append("stdout:\n" + stdout)
    if stderr:
        parts.append("stderr:\n" + stderr)
    parts.append(f"Para aceptar una compilacion sin OpenMP, define {ALLOW_NO_OPENMP_ENV}=1.")
    return "\n".join(parts)


def compile_c_target(
    source: str | Path,
    output: str | Path,
    *,
    target_kind: str,
    openmp: bool = True,
    logger: Callable[[str], None] | None = None,
) -> CompileResult:
    """Compile a C backend according to the repository parallel policy."""
    src = Path(source).resolve()
    out = Path(output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"No existe el archivo C: {src}")

    def log(message: str) -> None:
        if logger is not None:
            logger(message)

    try:
        cmd = build_c_compile_command(src, out, target_kind=target_kind, openmp=bool(openmp))
    except RuntimeError as exc:
        if not openmp or not allow_no_openmp():
            raise
        no_omp_cmd = build_c_compile_command(src, out, target_kind=target_kind, openmp=False)
        log(
            "No se pudo preparar OpenMP y ALLOW_NO_OPENMP=1 esta activo; "
            "compilando backend sin paralelismo OpenMP. Detalle: " + str(exc)
        )
        subprocess.run(no_omp_cmd, check=True)
        return CompileResult(
            path=out,
            command=no_omp_cmd,
            openmp_requested=True,
            openmp_active=False,
            compiler=no_omp_cmd[0],
            target_kind=target_kind,
        )
    if not openmp:
        log("OpenMP deshabilitado por configuracion; backend sin paralelismo OpenMP.")
    log("Compilando C: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=bool(openmp), text=True)
        return CompileResult(
            path=out,
            command=cmd,
            openmp_requested=bool(openmp),
            openmp_active=bool(openmp),
            compiler=cmd[0],
            target_kind=target_kind,
        )
    except subprocess.CalledProcessError as exc:
        if not openmp:
            raise RuntimeError(_format_compile_failure(cmd, exc)) from exc
        if not allow_no_openmp():
            raise RuntimeError(_format_compile_failure(cmd, exc)) from exc

        no_omp_cmd = build_c_compile_command(src, out, target_kind=target_kind, openmp=False)
        log(
            "OpenMP fallo y ALLOW_NO_OPENMP=1 esta activo; "
            "compilando backend sin paralelismo OpenMP."
        )
        subprocess.run(no_omp_cmd, check=True)
        return CompileResult(
            path=out,
            command=no_omp_cmd,
            openmp_requested=True,
            openmp_active=False,
            compiler=no_omp_cmd[0],
            target_kind=target_kind,
        )


def parallel_contract(
    *,
    python_workers: int,
    omp_threads: int,
    backend_openmp_active: bool,
    seed_strategy: str = "not_applicable",
    stage_kind: str,
) -> Dict[str, object]:
    """Small serializable contract for logs and JSON summaries."""
    return {
        "python_workers": max(1, int(python_workers)),
        "omp_threads": max(1, int(omp_threads)),
        "backend_openmp_active": bool(backend_openmp_active),
        "seed_strategy": str(seed_strategy),
        "stage_kind": str(stage_kind),
    }
