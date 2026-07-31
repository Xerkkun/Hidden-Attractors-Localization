"""Build and smoke-test the PyPI wheel in a temporary environment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_SDIST_FILES = {
    "LICENSE",
    "MANIFEST.in",
    "MANIFEST.md",
    "PKG-INFO",
    "README.md",
    "USER_MANUAL.md",
    "docs",
    "docs/assets",
    "docs/javascripts",
    "examples",
    "figure_scripts",
    "pyproject.toml",
    "setup.cfg",
    "examples/minimal_chua_protocol.py",
    "examples/quickstart_equilibria.py",
    "docs/api_stability.md",
    "docs/citation.md",
    "docs/installation.md",
    "docs/quick_start.md",
    "docs/scientific_scope.md",
    "docs/mathematical_diagnostics.md",
    "docs/plot_function_catalog.md",
    "docs/javascripts/mathjax.js",
    "figure_scripts/generate_plot_catalog_examples.py",
}
ALLOWED_SDIST_PREFIXES = (
    "hidden_attractors/",
    "hidden_attractors_fo.egg-info/",
    "examples/chua_integer_lure_reference/",
    "docs/assets/generated_plot_catalog/",
)


def _run(
    cmd: list[str | Path],
    *,
    cwd: Path = VERSION_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(str(part) for part in cmd)
    print(f"$ {printable}")
    result = subprocess.run(
        [str(part) for part in cmd],
        cwd=cwd,
        env=env,
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


def _validate_wheel_contents(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = {name.replace("\\", "/") for name in archive.namelist()}

    if "hidden_attractors/__init__.py" not in names:
        raise RuntimeError("wheel does not contain hidden_attractors/__init__.py")
    if not any(name.startswith("hidden_attractors/native/csrc/") for name in names):
        raise RuntimeError("wheel does not contain the native C sources")

    unexpected = sorted(
        name
        for name in names
        if not name.startswith("hidden_attractors/")
        and ".dist-info/" not in name
    )
    if unexpected:
        raise RuntimeError(f"wheel contains unexpected public paths: {unexpected[:20]}")


def _validate_sdist_contents(sdist: Path) -> None:
    with tarfile.open(sdist, mode="r:gz") as archive:
        raw_names = [name.replace("\\", "/") for name in archive.getnames()]

    roots = {name.split("/", maxsplit=1)[0] for name in raw_names if name}
    if len(roots) != 1:
        raise RuntimeError(f"sdist must have one archive root, found: {sorted(roots)}")
    root = next(iter(roots))
    names = {
        name.removeprefix(root + "/")
        for name in raw_names
        if name != root
    }

    required = {
        "README.md",
        "USER_MANUAL.md",
        "hidden_attractors/__init__.py",
        "examples/chua_integer_lure_reference/run_example.py",
        "docs/installation.md",
        "docs/quick_start.md",
        "docs/mathematical_diagnostics.md",
        "docs/plot_function_catalog.md",
        "docs/assets/generated_plot_catalog/catalog_results.json",
        "figure_scripts/generate_plot_catalog_examples.py",
    }
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"sdist is missing required public paths: {missing}")

    unexpected = sorted(
        name
        for name in names
        if name not in ALLOWED_SDIST_FILES
        and not any(
            name == prefix.rstrip("/") or name.startswith(prefix)
            for prefix in ALLOWED_SDIST_PREFIXES
        )
    )
    if unexpected:
        raise RuntimeError(f"sdist contains unexpected public paths: {unexpected[:20]}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate distribution contents and a clean wheel installation."
    )
    parser.add_argument(
        "--use-existing-dist",
        action="store_true",
        help="validate the existing dist/ artifacts instead of rebuilding them",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="extract and smoke-test the wheel with the current environment's dependencies",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not args.use_existing_dist:
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
        _validate_wheel_contents(wheels[0])
        _validate_sdist_contents(sdists[0])

        if args.offline:
            with tempfile.TemporaryDirectory(prefix="ha-wheel-test-") as tmp:
                tmp_path = Path(tmp)
                site_dir = tmp_path / "wheel-site"
                with zipfile.ZipFile(wheels[0]) as archive:
                    archive.extractall(site_dir)
                env = os.environ.copy()
                env["PYTHONPATH"] = str(site_dir)
                location_check = (
                    "from pathlib import Path; import hidden_attractors; "
                    f"root=Path({str(site_dir)!r}).resolve(); "
                    "loaded=Path(hidden_attractors.__file__).resolve(); "
                    "assert loaded.is_relative_to(root), (loaded, root); "
                    "print(f'import ok from {loaded}')"
                )
                _run([sys.executable, "-c", location_check], cwd=tmp_path, env=env)
                _run(
                    [sys.executable, "-m", "hidden_attractors.cli.main", "--help"],
                    cwd=tmp_path,
                    env=env,
                )
                seed_help = _run(
                    [sys.executable, "-m", "hidden_attractors.cli.main", "seed", "--help"],
                    cwd=tmp_path,
                    env=env,
                )
                help_text = seed_help.stdout.lower()
                if "machado" in help_text or "fdf" in help_text:
                    raise RuntimeError("Machado/FDF appeared in public seed help")
        else:
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
