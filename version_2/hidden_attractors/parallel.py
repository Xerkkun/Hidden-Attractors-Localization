#!/usr/bin/env python3
"""Shared parallelism policy for the fractional Chua backends.

Stability: internal
    Compilation helpers, OpenMP flags, and process-pool policy.  These are
    consumed by workflows and backends and are not a public compatibility
    surface.

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
import hashlib
import ctypes
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterator, List, MutableMapping, Sequence


ALLOW_NO_OPENMP_ENV = "ALLOW_NO_OPENMP"
_BUILD_LOCKS_GUARD = threading.Lock()
_BUILD_LOCKS: dict[Path, threading.RLock] = {}
_LOCAL_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
_CTYPES_ABI_LAYOUT_VERSION = "hafo-ctypes-abi-v3"


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
        # Keep shared libraries loadable from Python even when the MSYS2 bin
        # directory is not inherited by the DLL loader.
        return compiler, flag, ["-static-libgcc"]

    compiler = os.environ.get("CC", "gcc")
    flag = ["-fopenmp"] if openmp else []
    return compiler, flag, []


def build_c_compile_command(source: Path, output: Path, *, target_kind: str, openmp: bool) -> List[str]:
    compiler, cflags, ldflags = _compiler_and_flags(openmp, target_kind)
    cmd = [compiler, "-O3", "-std=c11"]
    if target_kind == "shared":
        cmd.append("-shared")
        if platform.system().lower() != "windows":
            cmd.append("-fPIC")
    cmd.extend(cflags)
    cmd.extend(["-o", str(output), str(source), "-lm"])
    cmd.extend(ldflags)
    return cmd


def _local_dependencies(source: Path) -> tuple[Path, ...]:
    """Return the source and recursively included local headers."""

    pending = [source.resolve()]
    seen: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            text = current.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = current.read_text(encoding="utf-8", errors="replace")
        for relative in _LOCAL_INCLUDE_RE.findall(text):
            dependency = (current.parent / relative).resolve()
            if not dependency.is_file():
                raise FileNotFoundError(
                    f"No existe la dependencia C local {relative!r} incluida por {current}."
                )
            pending.append(dependency)
    return tuple(sorted(seen, key=lambda path: str(path).casefold()))


@lru_cache(maxsize=16)
def _compiler_identity(compiler: str) -> str:
    resolved = shutil.which(compiler) or compiler
    try:
        completed = subprocess.run(
            [compiler, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = (completed.stdout or completed.stderr or "").splitlines()
        first_line = version[0].strip() if version else "unknown-version"
    except (OSError, subprocess.SubprocessError):
        first_line = "unknown-version"
    return f"{Path(resolved).resolve() if Path(resolved).exists() else resolved}|{first_line}"


def _native_abi_fingerprint() -> str:
    """Return platform fields that determine native/ctypes compatibility."""

    return "|".join(
        (
            _CTYPES_ABI_LAYOUT_VERSION,
            sys.platform,
            platform.system(),
            platform.machine(),
            f"pointer-bits={struct.calcsize('P') * 8}",
            f"byteorder={sys.byteorder}",
        )
    )


def _content_addressed_output(
    source: Path,
    requested_output: Path,
    *,
    target_kind: str,
    openmp: bool,
) -> tuple[Path, List[str]]:
    """Return an immutable artifact path derived from all compilation inputs."""

    template = build_c_compile_command(
        source,
        Path("__HAFO_NATIVE_OUTPUT__"),
        target_kind=target_kind,
        openmp=openmp,
    )
    digest = hashlib.sha256()
    digest.update(b"hafo-native-build-v3\0")
    digest.update(target_kind.encode("utf-8") + b"\0")
    digest.update(str(bool(openmp)).encode("ascii") + b"\0")
    digest.update(_native_abi_fingerprint().encode("utf-8") + b"\0")
    digest.update(_compiler_identity(template[0]).encode("utf-8") + b"\0")
    for argument in template:
        digest.update(str(argument).encode("utf-8") + b"\0")
    for dependency in _local_dependencies(source):
        digest.update(str(dependency).encode("utf-8") + b"\0")
        digest.update(dependency.read_bytes())
        digest.update(b"\0")
    short_hash = digest.hexdigest()[:20]
    suffix = requested_output.suffix
    stem = requested_output.name[: -len(suffix)] if suffix else requested_output.name
    artifact = requested_output.with_name(f"{stem}-{short_hash}{suffix}")
    command = build_c_compile_command(
        source,
        artifact,
        target_kind=target_kind,
        openmp=openmp,
    )
    return artifact, command


def _thread_lock_for(path: Path) -> threading.RLock:
    with _BUILD_LOCKS_GUARD:
        return _BUILD_LOCKS.setdefault(path, threading.RLock())


def _artifact_digest_path(artifact: Path) -> Path:
    return artifact.with_name(artifact.name + ".sha256")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_digest_is_valid(artifact: Path) -> bool:
    sidecar = _artifact_digest_path(artifact)
    if not artifact.is_file() or not sidecar.is_file():
        return False
    try:
        expected = sidecar.read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            return False
        return _sha256_file(artifact) == expected
    except (OSError, UnicodeError):
        return False


@contextmanager
def _interprocess_build_lock(path: Path, timeout: float = 120.0) -> Iterator[None]:
    """Serialize publication of one immutable artifact across processes."""

    local_lock = _thread_lock_for(path)
    with local_lock:
        lock_path = path.with_name(path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            deadline = time.monotonic() + timeout
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timeout esperando el bloqueo de compilacion {lock_path}.")
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _compile_immutable_artifact(
    source: Path,
    requested_output: Path,
    *,
    target_kind: str,
    openmp: bool,
    logger: Callable[[str], None] | None,
) -> tuple[Path, List[str]]:
    artifact, command = _content_addressed_output(
        source,
        requested_output,
        target_kind=target_kind,
        openmp=openmp,
    )
    if _artifact_digest_is_valid(artifact):
        return artifact, command

    with _interprocess_build_lock(artifact):
        if _artifact_digest_is_valid(artifact):
            return artifact, command
        sidecar = _artifact_digest_path(artifact)
        artifact.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        token = f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
        temp_suffix = artifact.suffix
        temp_stem = artifact.name[: -len(temp_suffix)] if temp_suffix else artifact.name
        temporary = artifact.with_name(f".{temp_stem}.{token}.tmp{temp_suffix}")
        temporary_sidecar = artifact.with_name(f".{temp_stem}.{token}.tmp.sha256")
        temp_command = build_c_compile_command(
            source,
            temporary,
            target_kind=target_kind,
            openmp=openmp,
        )
        if logger is not None:
            logger("Compilando C: " + " ".join(temp_command))
        try:
            subprocess.run(
                temp_command,
                check=True,
                capture_output=True,
                text=True,
            )
            temporary_sidecar.write_text(
                _sha256_file(temporary) + "\n",
                encoding="ascii",
            )
            os.replace(temporary, artifact)
            os.replace(temporary_sidecar, sidecar)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            try:
                temporary_sidecar.unlink()
            except FileNotFoundError:
                pass
    return artifact, command


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
    """Compile an immutable, content-addressed C backend atomically."""
    src = Path(source).resolve()
    out = Path(output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"No existe el archivo C: {src}")

    def log(message: str) -> None:
        if logger is not None:
            logger(message)

    requested_openmp = bool(openmp)
    try:
        artifact, cmd = _compile_immutable_artifact(
            src,
            out,
            target_kind=target_kind,
            openmp=requested_openmp,
            logger=logger,
        )
    except subprocess.CalledProcessError as exc:
        failed_command = list(exc.cmd) if isinstance(exc.cmd, (list, tuple)) else [str(exc.cmd)]
        if not requested_openmp or not allow_no_openmp():
            raise RuntimeError(_format_compile_failure(failed_command, exc)) from exc
        log(
            "OpenMP fallo y ALLOW_NO_OPENMP=1 esta activo; "
            "compilando backend sin paralelismo OpenMP."
        )
        try:
            artifact, cmd = _compile_immutable_artifact(
                src,
                out,
                target_kind=target_kind,
                openmp=False,
                logger=logger,
            )
        except subprocess.CalledProcessError as fallback_exc:
            fallback_command = (
                list(fallback_exc.cmd)
                if isinstance(fallback_exc.cmd, (list, tuple))
                else [str(fallback_exc.cmd)]
            )
            raise RuntimeError(
                _format_compile_failure(fallback_command, fallback_exc)
            ) from fallback_exc
        return CompileResult(path=artifact, command=cmd, openmp_requested=True,
                             openmp_active=False, compiler=cmd[0], target_kind=target_kind)
    except RuntimeError as exc:
        if not openmp or not allow_no_openmp():
            raise
        log(
            "No se pudo preparar OpenMP y ALLOW_NO_OPENMP=1 esta activo; "
            "compilando backend sin paralelismo OpenMP. Detalle: " + str(exc)
        )
        try:
            artifact, cmd = _compile_immutable_artifact(
                src,
                out,
                target_kind=target_kind,
                openmp=False,
                logger=logger,
            )
        except subprocess.CalledProcessError as fallback_exc:
            fallback_command = (
                list(fallback_exc.cmd)
                if isinstance(fallback_exc.cmd, (list, tuple))
                else [str(fallback_exc.cmd)]
            )
            raise RuntimeError(
                _format_compile_failure(fallback_command, fallback_exc)
            ) from fallback_exc
        return CompileResult(path=artifact, command=cmd, openmp_requested=True,
                             openmp_active=False, compiler=cmd[0], target_kind=target_kind)
    if not openmp:
        log("OpenMP deshabilitado por configuracion; backend sin paralelismo OpenMP.")
    return CompileResult(
        path=artifact,
        command=cmd,
        openmp_requested=requested_openmp,
        openmp_active=requested_openmp,
        compiler=cmd[0],
        target_kind=target_kind,
    )


def load_ctypes_library(
    source: str | Path,
    output: str | Path,
    *,
    expected_symbols: Sequence[str],
    expected_abi_version: int | None = None,
    abi_version_symbol: str = "hafo_native_abi_version",
    openmp: bool = True,
    logger: Callable[[str], None] | None = None,
) -> tuple[CompileResult, ctypes.CDLL]:
    """Build and load a shared library, healing one poisoned cache entry.

    The content-addressed filename prevents ordinary stale reuse.  This extra
    load check handles a truncated or externally replaced file at that exact
    name.  Only the failed immutable artifact is removed, under its build
    lock, and one atomic rebuild is attempted.
    """

    required = tuple(str(symbol) for symbol in expected_symbols)
    if not required or any(not symbol for symbol in required):
        raise ValueError("expected_symbols must contain at least one symbol name.")

    result = compile_c_target(
        source,
        output,
        target_kind="shared",
        openmp=openmp,
        logger=logger,
    )
    first_error: BaseException | None = None
    for attempt in range(2):
        try:
            library = ctypes.CDLL(str(result.path.resolve()))
            missing = [symbol for symbol in required if not hasattr(library, symbol)]
            if missing:
                raise AttributeError(
                    "Native library is missing required symbols: " + ", ".join(missing)
                )
            if expected_abi_version is not None:
                version_function = getattr(library, abi_version_symbol)
                version_function.argtypes = []
                version_function.restype = ctypes.c_int
                actual_version = int(version_function())
                if actual_version != expected_abi_version:
                    raise RuntimeError(
                        f"Native ABI version {actual_version} does not match "
                        f"expected version {expected_abi_version}."
                    )
            return result, library
        except (OSError, AttributeError, RuntimeError) as exc:
            if attempt == 1:
                raise RuntimeError(
                    f"Native artifact {result.path} failed validation after rebuild: {exc}"
                ) from exc
            first_error = exc
            try:
                failed_stat = result.path.stat()
            except FileNotFoundError:
                failed_stat = None
            with _interprocess_build_lock(result.path):
                try:
                    current_stat = result.path.stat()
                except FileNotFoundError:
                    current_stat = None
                if current_stat is not None and (
                    failed_stat is None
                    or (
                        current_stat.st_size == failed_stat.st_size
                        and current_stat.st_mtime_ns == failed_stat.st_mtime_ns
                    )
                ):
                    result.path.unlink()
                    _artifact_digest_path(result.path).unlink(missing_ok=True)
            if logger is not None:
                logger(
                    f"Artefacto nativo invalido descartado ({first_error}); "
                    "recompilando una vez."
                )
            result = compile_c_target(
                source,
                output,
                target_kind="shared",
                openmp=openmp,
                logger=logger,
            )

    raise AssertionError("unreachable native library load state")


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
