"""Build and smoke-test the PyPI wheel in a temporary environment."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str | Path], *, cwd: Path = VERSION_ROOT) -> subprocess.CompletedProcess[str]:
    printable = " ".join(str(part) for part in cmd)
    print(f"$ {printable}")
    result = subprocess.run(
        [str(part) for part in cmd],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {printable}")
    return result


def _make_writable(path: Path) -> None:
    if not path.exists():
        return
    paths = sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True) if path.is_dir() else []
    for child in paths:
        try:
            child.chmod(0o700 if child.is_dir() else 0o600)
        except OSError:
            pass
    try:
        path.chmod(0o700 if path.is_dir() else 0o600)
    except OSError:
        pass


def _remove_path(path: Path) -> None:
    _make_writable(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _clean_build_artifacts() -> None:
    for name in ("dist", "build"):
        path = VERSION_ROOT / name
        if path.exists():
            _remove_path(path)
    for path in VERSION_ROOT.glob("*.egg-info"):
        _remove_path(path)


def _venv_python(env_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _venv_script(env_dir: Path, name: str) -> Path:
    if sys.platform.startswith("win"):
        exe = env_dir / "Scripts" / f"{name}.exe"
        if exe.exists():
            return exe
        return env_dir / "Scripts" / name
    return env_dir / "bin" / name


def main() -> int:
    try:
        _clean_build_artifacts()
        _run([sys.executable, "-m", "build"])

        dist_files = sorted((VERSION_ROOT / "dist").glob("*"))
        wheels = [path for path in dist_files if path.suffix == ".whl"]
        sdists = [path for path in dist_files if path.name.endswith(".tar.gz")]
        if len(wheels) != 1:
            raise RuntimeError(f"expected exactly one wheel, found {len(wheels)}: {wheels}")
        if len(sdists) != 1:
            raise RuntimeError(f"expected exactly one sdist, found {len(sdists)}: {sdists}")

        _run([sys.executable, "-m", "twine", "check", *dist_files])

        with tempfile.TemporaryDirectory(prefix="ha-wheel-test-") as tmp:
            env_dir = Path(tmp) / "venv"
            venv.EnvBuilder(with_pip=True).create(env_dir)
            py = _venv_python(env_dir)
            cli = _venv_script(env_dir, "hidden-attractors")

            _run([py, "-m", "pip", "install", "--upgrade", "pip"])
            _run([py, "-m", "pip", "install", wheels[0]])
            _run([cli, "--help"])
            seed_help = _run([cli, "seed", "--help"])
            help_text = seed_help.stdout.lower()
            if "machado" in help_text or "fdf" in help_text:
                raise RuntimeError("Machado/FDF appeared in public seed help")
            _run([py, "-c", "import hidden_attractors; print('import ok')"])

    except Exception as exc:
        print(f"validate_wheel_install failed: {exc}", file=sys.stderr)
        return 1

    print("validate_wheel_install passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())