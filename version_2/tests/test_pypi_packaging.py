from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent


def _pyproject() -> dict:
    with (VERSION_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_pypi_project_metadata_and_public_cli() -> None:
    data = _pyproject()
    project = data["project"]

    assert project["name"] == "hidden-attractors-fo"
    assert project["version"] == "1.2.0"
    assert project["readme"] == "README.md"
    assert project["requires-python"] == ">=3.11,<3.15"
    assert project["license"] == "MIT"
    assert project["authors"]
    assert project["keywords"]
    assert project["classifiers"]
    assert project["urls"]["Repository"] == "https://github.com/Xerkkun/Hidden-Attractors-Localization"
    assert project["urls"]["Archive"] == "https://doi.org/10.17605/OSF.IO/ZGK74"

    scripts = project.get("scripts", {})
    assert scripts == {"hidden-attractors": "hidden_attractors.cli.main:main"}

    serialized_scripts = "\n".join(f"{key}={value}" for key, value in scripts.items()).lower()
    for blocked in ("machado", "fdf", "hidden-attractors-check-validation", "hidden-attractors-protocol"):
        assert blocked not in serialized_scripts


def test_pypi_readme_and_release_files_exist() -> None:
    readme = (VERSION_ROOT / "README.md").read_text(encoding="utf-8")
    readme_lower = re.sub(r"\s+", " ", readme.lower())

    assert "python -m pip install hidden-attractors-fo" in readme
    assert "import hidden_attractors" in readme
    assert "not a global proof" in readme_lower
    assert "finite numerical evidence" in readme_lower
    assert "not in the installed package" in readme_lower

    assert (VERSION_ROOT / "MANIFEST.in").exists()
    assert (VERSION_ROOT / "MANIFEST.md").exists()
    assert (VERSION_ROOT / "release_package" / "README_RELEASE.md").exists()
    assert (VERSION_ROOT / "release_package" / "PUBLISHING_POLICY.md").exists()
    assert (REPO_ROOT / ".github" / "workflows" / "package.yml").exists()
    assert (REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml").exists()


def test_pypi_wheel_package_scope_is_narrow() -> None:
    data = _pyproject()
    find = data["tool"]["setuptools"]["packages"]["find"]
    package_data = data["tool"]["setuptools"]["package-data"]

    assert find["include"] == ["hidden_attractors*"]
    assert "tools*" in find["exclude"]
    assert "benchmarks*" in find["exclude"]
    assert "hidden_attractors.native.tests*" in find["exclude"]
    assert "hidden_attractors" in package_data
    assert "native/csrc/*.c" in package_data["hidden_attractors"]
    assert "native/csrc/*.h" in package_data["hidden_attractors"]
    assert "configs/examples/workflow_contract.yaml" in package_data["hidden_attractors"]
    assert "configs/examples/*.yaml" not in package_data["hidden_attractors"]

    manifest = (VERSION_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include examples/chua_integer_lure_reference" in manifest
    assert "include hidden_attractors/configs/examples/workflow_contract.yaml" in manifest
    assert "recursive-include hidden_attractors *.py *.yaml" not in manifest
    assert "recursive-exclude hidden_attractors/native/tests *" in manifest
    assert "recursive-exclude tests *" in manifest
    assert not any(
        line.strip().startswith("prune ")
        for line in manifest.splitlines()
    )


@pytest.mark.packaging
@pytest.mark.release_readiness
def test_release_build_command_succeeds_when_enabled() -> None:
    if os.environ.get("RUN_PYPI_BUILD_TEST") != "1":
        pytest.skip("set RUN_PYPI_BUILD_TEST=1 to run python -m build inside pytest")

    result = subprocess.run(
        [sys.executable, "-m", "build"],
        cwd=VERSION_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
