"""Repository-only release-readiness checks.

This module is deliberately outside the importable PyPI package. It requires
the complete tagged repository, including release metadata and workflows.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Sequence


MOJIBAKE_PATTERNS = [
    "\u00c3\u0192",   # bad sequence marker
    "\u00c3\u201a",   # bad sequence marker
    "\u00c3",
    "\u00c2",
    "\u00e2\u20ac",
    "\u00e2\u20ac\u201d",
    "\u00e2\u20ac\u2122",
    "\u00e2\u20ac\u0153",
    "\u00e2\u20ac\u009d",
    "\u00e2\u201d",
]

MAIN_TEXT_PATTERNS = [
    "README.md",
    "CHANGELOG.md",
    "RELEASE_NOTES.md",
    "REPRODUCIBILITY.md",
    "CITATION.cff",
    ".zenodo.json",
    "codemeta.json",
    "version_2/README.md",
    "version_2/USER_MANUAL.md",
    "version_2/MANIFEST.md",
    "version_2/pyproject.toml",
    "version_2/docs/*.md",
    "version_2/release_package/*.md",
    "version_2/release_package/*.json",
    "version_2/release_package/sample_input/*.yaml",
    "version_2/release_package/sample_input/*.md",
    "version_2/release_package/sample_output/*.json",
    "version_2/release_package/sample_output/*.md",
]

PROMOTED_SCAN_PATTERNS = [
    "version_2/validation/**/*.json",
    "version_2/validation/**/*.md",
    "version_2/docs/**/*.md",
    "version_2/release_package/**/*.md",
    "version_2/release_package/**/*.json",
]

PUBLIC_SDIST_MANIFEST_DIRECTIVES = {
    "include README.md",
    "include LICENSE",
    "include pyproject.toml",
    "include MANIFEST.in",
    "include MANIFEST.md",
    "include USER_MANUAL.md",
    "recursive-include hidden_attractors *.py *.c *.h",
    "recursive-exclude hidden_attractors/native/tests *",
    "include hidden_attractors/configs/examples/workflow_contract.yaml",
    "recursive-include examples/chua_integer_lure_reference *.py *.md *.yaml",
    "include examples/quickstart_equilibria.py",
    "include examples/minimal_chua_protocol.py",
    "include docs/installation.md",
    "include docs/quick_start.md",
    "include docs/api_stability.md",
    "include docs/scientific_scope.md",
    "include docs/citation.md",
    "include docs/mathematical_diagnostics.md",
    "include docs/plot_function_catalog.md",
    "include docs/javascripts/mathjax.js",
    "recursive-include docs/assets/generated_plot_catalog *.png *.json",
    "include figure_scripts/generate_plot_catalog_examples.py",
    "recursive-exclude tests *",
    "global-exclude __pycache__ *.py[cod] .DS_Store",
}

PUBLICATION_STATE_PAIRS = {
    ("verified_release_candidate", "not_published"),
    ("published", "published"),
}

JSON_POLICY_KEYS = {
    "legacy_provenance",
    "archived_external_paths",
    "legacy_external_figures_not_promoted",
    "excluded_paths",
    "unpromoted_outputs",
}

LOCAL_PATH_REGEXES = [
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]"),
    re.compile(r"(^|[^A-Za-z0-9_])[\\/]Users[\\/]"),
    re.compile(r"(^|[^A-Za-z0-9_])/home/"),
    re.compile(r"(^|[\\/])Desktop([\\/]|$)"),
    re.compile(r"(^|[\\/])Downloads([\\/]|$)"),
    re.compile(r"OneDrive"),
    re.compile(r"Google Drive"),
]

VALIDATION_OUTPUTS_REGEX = re.compile(r"(^|[\\/])validation_outputs([\\/]|$)|version_2[\\/]validation_outputs")
PROJECT_NAME_PATH_REGEX = re.compile(r"Hidden Attractors Fractional Order[\\/]")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)


def _git_ls_files(root: Path, *patterns: str) -> list[str]:
    result = _git(root, "ls-files", *patterns)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def _git_head(root: Path, short: bool = False) -> str:
    args = ["rev-parse", "--short", "HEAD"] if short else ["rev-parse", "HEAD"]
    result = _git(root, *args)
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def _git_recent_commits(root: Path, limit: int = 5) -> list[str]:
    result = _git(root, "log", "-n", str(limit), "--format=%H")
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _git_recent_commits_short(root: Path, limit: int = 5) -> list[str]:
    result = _git(root, "log", "-n", str(limit), "--format=%h")
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _check(name: str, category: str, ok: bool, details: list[str] | None = None) -> dict[str, Any]:
    return {"name": name, "category": category, "ok": ok, "details": details or []}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"_load_error": str(exc)}

def _load_pyproject(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def _mojibake_hits(root: Path) -> list[str]:
    hits: list[str] = []
    seen: set[Path] = set()
    for pattern in MAIN_TEXT_PATTERNS:
        for path in root.glob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError as exc:
                hits.append(f"{path.relative_to(root).as_posix()}: utf-8 decode failed: {exc}")
                continue
            for bad in MOJIBAKE_PATTERNS:
                if bad in text:
                    hits.append(f"{path.relative_to(root).as_posix()}: contains {bad!r}")
                    break
    return hits


def _manifest_path_references(root: Path, manifest: dict[str, Any], key: str) -> list[str]:
    values = manifest.get(key, [])
    if not isinstance(values, list):
        return [f"{key} is not a list"]
    missing = []
    for rel in values:
        if not isinstance(rel, str) or not (root / rel).exists():
            missing.append(str(rel))
    return missing


def _is_policy_markdown_line(lines: list[str], index: int) -> bool:
    current_header = ""
    for previous in lines[: index + 1]:
        if previous.startswith("#"):
            current_header = previous.lower()
    line = lines[index].lower()
    policy_terms = [
        "policy",
        "evidence boundary",
        "local/regenerable",
        "local outputs",
        "unpromoted",
        "non-promoted",
        "legacy",
        "freeze audit",
        "ci and freeze",
    ]
    return any(term in current_header or term in line for term in policy_terms)


def _string_path_violation(value: str, *, allow_validation_outputs: bool) -> bool:
    if any(regex.search(value) for regex in LOCAL_PATH_REGEXES):
        return True
    if PROJECT_NAME_PATH_REGEX.search(value):
        return True
    if VALIDATION_OUTPUTS_REGEX.search(value) and not allow_validation_outputs:
        return True
    return False


def _json_path_hits(path: Path, root: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return [f"{path.relative_to(root).as_posix()}: JSON parse failed: {exc}"]
    hits: list[str] = []

    def walk(value: Any, keys: tuple[str, ...] = (), policy_context: bool = False) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_policy = policy_context or key in JSON_POLICY_KEYS
                walk(child, (*keys, str(key)), child_policy)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                walk(child, (*keys, str(idx)), policy_context)
        elif isinstance(value, str):
            if _string_path_violation(value, allow_validation_outputs=policy_context):
                dotted = ".".join(keys) or "<root>"
                hits.append(f"{path.relative_to(root).as_posix()}:{dotted}: {value}")

    walk(data)
    return hits


def _text_path_hits(path: Path, root: Path) -> list[str]:
    hits = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        return [f"{path.relative_to(root).as_posix()}: utf-8 decode failed: {exc}"]
    for idx, line in enumerate(lines):
        allow_validation_outputs = path.suffix.lower() == ".md" and _is_policy_markdown_line(lines, idx)
        if _string_path_violation(line, allow_validation_outputs=allow_validation_outputs):
            hits.append(f"{path.relative_to(root).as_posix()}:L{idx + 1}: {line.strip()}")
    return hits


def _promoted_local_path_hits(root: Path) -> list[str]:
    hits: list[str] = []
    seen: set[Path] = set()
    eligible_result = _git(
        root,
        "ls-files",
        "--cached",
    )
    if eligible_result.returncode != 0:
        raise RuntimeError(eligible_result.stderr.strip() or "git ls-files failed")
    eligible = {
        (root / relative.strip()).resolve()
        for relative in eligible_result.stdout.splitlines()
        if relative.strip()
    }
    for pattern in PROMOTED_SCAN_PATTERNS:
        for path in root.glob(pattern):
            if "outputs/wolfram" in path.as_posix():
                continue
            if not path.is_file() or path.resolve() not in eligible or path in seen:
                continue
            seen.add(path)
            if path.suffix.lower() == ".json":
                hits.extend(_json_path_hits(path, root))
            elif path.suffix.lower() in {".md", ".tex", ".bib"}:
                hits.extend(_text_path_hits(path, root))
    return hits


def _pypi_readiness_checks(root: Path, version_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    pyproject_path = version_root / "pyproject.toml"
    pyproject = _load_pyproject(pyproject_path)
    project = pyproject.get("project", {}) if isinstance(pyproject, dict) else {}
    project_version = str(project.get("version", ""))
    tool = pyproject.get("tool", {}) if isinstance(pyproject, dict) else {}
    setuptools = tool.get("setuptools", {}) if isinstance(tool, dict) else {}
    find_config = setuptools.get("packages", {}).get("find", {}) if isinstance(setuptools, dict) else {}
    package_data = setuptools.get("package-data", {}) if isinstance(setuptools, dict) else {}

    metadata_details: list[str] = []
    if pyproject.get("_load_error"):
        metadata_details.append(f"pyproject load error: {pyproject['_load_error']}")
    if project.get("name") != "hidden-attractors-fo":
        metadata_details.append(f"project.name={project.get('name')}")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", project_version):
        metadata_details.append(f"project.version is not a stable semantic version: {project_version}")
    if project.get("readme") != "README.md":
        metadata_details.append(f"project.readme={project.get('readme')}")
    if project.get("requires-python") != ">=3.11,<3.15":
        metadata_details.append(f"requires-python={project.get('requires-python')}")
    if project.get("license") != "MIT":
        metadata_details.append(f"license={project.get('license')}")
    if "LICENSE" not in project.get("license-files", []):
        metadata_details.append("LICENSE missing from license-files")
    if not project.get("authors"):
        metadata_details.append("authors missing")
    if not project.get("keywords"):
        metadata_details.append("keywords missing")
    classifiers = project.get("classifiers", [])
    if not classifiers:
        metadata_details.append("classifiers missing")
    dependencies = project.get("dependencies", [])
    for dependency in ("numpy", "matplotlib", "scipy", "numba", "PyYAML"):
        if not any(str(item).lower().startswith(dependency.lower()) for item in dependencies):
            metadata_details.append(f"runtime dependency missing: {dependency}")
    scripts = project.get("scripts", {})
    if scripts != {"hidden-attractors": "hidden_attractors.cli.main:main"}:
        metadata_details.append(f"project.scripts={scripts}")
    urls = project.get("urls", {})
    for key in ("Homepage", "Documentation", "Repository", "Issues", "Archive"):
        if key not in urls:
            metadata_details.append(f"project.urls missing {key}")
    if find_config.get("include") != ["hidden_attractors*"]:
        metadata_details.append(f"package include={find_config.get('include')}")
    excluded = set(find_config.get("exclude", []))
    for pattern in (
        "tools*",
        "benchmarks*",
        "tests*",
        "examples*",
        "hidden_attractors.native.tests*",
    ):
        if pattern not in excluded:
            metadata_details.append(f"package exclude missing {pattern}")
    hidden_data = package_data.get("hidden_attractors", [])
    for pattern in (
        "native/csrc/*.c",
        "native/csrc/*.h",
        "configs/examples/workflow_contract.yaml",
    ):
        if pattern not in hidden_data:
            metadata_details.append(f"package-data missing {pattern}")
    if any(key.startswith("tools") for key in package_data):
        metadata_details.append("tools package-data must not be in the wheel")
    checks.append(_check("PyPI project metadata", "software", not metadata_details, metadata_details))

    readme_path = version_root / "README.md"
    readme = readme_path.read_text(encoding="utf-8-sig") if readme_path.exists() else ""
    readme_lower = re.sub(r"\s+", " ", readme.lower())
    readme_details = []
    for required_text in (
        "python -m pip install hidden-attractors-fo",
        "import hidden_attractors",
        "hidden-attractors --help",
        "hidden-attractors inspect systems",
        "10.17605/OSF.IO/ZGK74",
    ):
        if required_text not in readme:
            readme_details.append(f"README missing {required_text}")
    for required_lower in ("not a global proof", "finite numerical evidence", "not in the installed package"):
        if required_lower not in readme_lower:
            readme_details.append(f"README missing {required_lower}")
    checks.append(_check("PyPI README", "software", not readme_details, readme_details))

    manifest_in = version_root / "MANIFEST.in"
    manifest_details = []
    if not manifest_in.exists():
        manifest_details.append("version_2/MANIFEST.in missing")
    else:
        manifest_text = manifest_in.read_text(encoding="utf-8-sig")
        actual_directives = {
            line.strip()
            for line in manifest_text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        missing_directives = sorted(
            PUBLIC_SDIST_MANIFEST_DIRECTIVES - actual_directives
        )
        unexpected_directives = sorted(
            actual_directives - PUBLIC_SDIST_MANIFEST_DIRECTIVES
        )
        manifest_details.extend(
            f"MANIFEST.in missing directive: {directive}"
            for directive in missing_directives
        )
        manifest_details.extend(
            f"MANIFEST.in unexpected directive: {directive}"
            for directive in unexpected_directives
        )
    checks.append(_check("PyPI MANIFEST.in", "software", not manifest_details, manifest_details))

    workflow_path = root / ".github" / "workflows" / "package.yml"
    workflow_details = []
    if not workflow_path.exists():
        workflow_details.append(".github/workflows/package.yml missing")
    else:
        workflow_text = workflow_path.read_text(encoding="utf-8-sig")
        for required_text in (
            "python -m build",
            "python -m twine check dist/*",
            "python -m pip install dist/*.whl",
            "hidden-attractors --help",
            "hidden-attractors seed --help",
            "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
        ):
            if required_text not in workflow_text:
                workflow_details.append(f"package.yml missing {required_text}")
    checks.append(_check("PyPI package workflow", "software", not workflow_details, workflow_details))

    publish_workflow = root / ".github" / "workflows" / "publish-pypi.yml"
    publish_details = []
    if not publish_workflow.exists():
        publish_details.append(".github/workflows/publish-pypi.yml missing")
    else:
        publish_text = publish_workflow.read_text(encoding="utf-8-sig")
        for required_text in (
            "Require an exact version-matched tag and commit",
            'if not ref.startswith("refs/tags/")',
            'tag != f"v{version}"',
            'git rev-parse "${tag}^{commit}"',
            'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"',
            'python -m pytest -q -m "not slow"',
            "python -m build",
            "python -m twine check dist/*",
            "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
            "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093",
            "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
            "environment:",
            "name: pypi",
        ):
            if required_text not in publish_text:
                publish_details.append(f"publish-pypi.yml missing {required_text}")
        if publish_text.count("id-token: write") != 1:
            publish_details.append("publish-pypi.yml must grant id-token: write exactly once")
        publish_job = publish_text.split("\n  publish:", maxsplit=1)
        if len(publish_job) != 2 or "id-token: write" not in publish_job[1]:
            publish_details.append("OIDC permission must be limited to the publish job")
        if "id-token: write" in publish_job[0]:
            publish_details.append("verification workflow must not receive OIDC permission")
    checks.append(_check("PyPI publish workflow", "software", not publish_details, publish_details))

    public_details = []
    main_text = (version_root / "hidden_attractors" / "cli" / "main.py").read_text(encoding="utf-8-sig")
    if '"seed": ["lure-centered", "lure-biased"]' not in main_text:
        public_details.append("seed public commands are not limited to lure-centered/lure-biased")
    if "machado" in str(project.get("scripts", {})).lower() or "fdf" in str(project.get("scripts", {})).lower():
        public_details.append("Machado/FDF appears in project.scripts")
    checks.append(_check("PyPI public CLI scope", "software", not public_details, public_details))

    pypi = manifest.get("pypi_readiness", {})
    expected = {
        "package_name": "hidden-attractors-fo",
        "import_name": "hidden_attractors",
        "version": project_version,
        "target_version": project_version,
        "build_backend": "setuptools.build_meta",
        "local_release_candidate_verification": "passed",
        "pypi_url": "https://pypi.org/project/hidden-attractors-fo/",
        "workflow": ".github/workflows/publish-pypi.yml",
        "authentication": "trusted_publishing_oidc",
        "publication_gate": "pypi_environment",
    }
    pypi_details = []
    if not isinstance(pypi, dict):
        pypi_details.append("archive manifest pypi_readiness missing or not an object")
    else:
        for key, value in expected.items():
            if pypi.get(key) != value:
                pypi_details.append(f"pypi_readiness.{key}={pypi.get(key)}")
        manifest_publication = manifest.get("publication_status")
        pypi_publication = pypi.get("publication_status")
        release_state = manifest.get("release_state")
        if pypi_publication != manifest_publication:
            pypi_details.append(
                "pypi_readiness.publication_status must match publication_status"
            )
        if (release_state, manifest_publication) not in PUBLICATION_STATE_PAIRS:
            pypi_details.append(
                "release_state/publication_status is not a closed coherent state"
            )
        current_public_version = pypi.get("current_public_version")
        if manifest_publication == "not_published":
            if not current_public_version or current_public_version == project_version:
                pypi_details.append(
                    "prepublication current_public_version must identify an earlier release"
                )
        elif manifest_publication == "published":
            if current_public_version != project_version:
                pypi_details.append(
                    "published current_public_version must equal the package version"
                )
    checks.append(_check("archive manifest PyPI readiness", "software", not pypi_details, pypi_details))

    return checks

def validate_release_readiness(argv: Sequence[str] | None = None) -> None:
    """Validate release repository/software readiness without changing science artifacts."""
    parser = argparse.ArgumentParser(description="Validate release readiness metadata and hygiene")
    parser.add_argument("--json", action="store_true", help="Output machine-readable results")
    parser.add_argument("--strict", action="store_true", help="Fail on correctable repository/software readiness errors")
    parser.add_argument(
        "--submission-strict",
        action="store_true",
        help="Backward-compatible alias for strict repository verification",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    version_root = root / "version_2"
    release_root = version_root / "release_package"
    manifest_path = release_root / "archive_manifest.json"
    manifest = _load_json(manifest_path)
    checks: list[dict[str, Any]] = []

    required = [
        "CITATION.cff",
        ".zenodo.json",
        "codemeta.json",
        "AUTHORS.md",
        "CHANGELOG.md",
        "RELEASE_NOTES.md",
        "REPRODUCIBILITY.md",
        "version_2/release_package/README_RELEASE.md",
        "version_2/release_package/PROGRAM_SUMMARY.md",
        "version_2/release_package/SAMPLE_RUN.md",
        "version_2/release_package/PUBLISHING_POLICY.md",
        "version_2/MANIFEST.in",
        "version_2/MANIFEST.md",
        ".github/workflows/package.yml",
        ".github/workflows/publish-pypi.yml",
        "version_2/release_package/archive_manifest.json",
        "version_2/release_package/sample_input/chua_integer_comprehensive.yaml",
        "version_2/release_package/sample_output/comprehensive_sample_summary.json",
        "version_2/README.md",
        "version_2/USER_MANUAL.md",
        "version_2/docs/manual_manifest.yaml",
    ]
    missing = [rel for rel in required if not (root / rel).exists()]
    checks.append(_check("required release metadata", "repository", not missing, missing))

    validation_outputs_tracked = _git_ls_files(root, "version_2/validation_outputs")
    checks.append(_check("validation_outputs untracked", "repository", not validation_outputs_tracked, validation_outputs_tracked))

    path_hits = _promoted_local_path_hits(root)
    checks.append(_check("no local absolute paths in promoted evidence", "repository", not path_hits, path_hits[:50]))

    sample_input = release_root / "sample_input"
    sample_output = release_root / "sample_output"
    sample_details = []
    sample_config = sample_input / "chua_integer_comprehensive.yaml"
    sample_summary_path = sample_output / "comprehensive_sample_summary.json"
    for expected_path in (
        sample_input / "README.md",
        sample_config,
        sample_output / "README.md",
        sample_summary_path,
    ):
        if not expected_path.exists():
            sample_details.append(f"{expected_path.relative_to(release_root).as_posix()} missing")
    sample_details.extend(_manifest_path_references(root, manifest, "sample_input"))
    sample_details.extend(_manifest_path_references(root, manifest, "sample_output"))
    if manifest.get("sample_status") != "executed":
        sample_details.append(f"archive manifest sample_status={manifest.get('sample_status')}")
    sample_summary = _load_json(sample_summary_path) if sample_summary_path.exists() else {}
    project_version = str(_load_pyproject(version_root / "pyproject.toml").get("project", {}).get("version", ""))
    expected_summary_values = {
        "sample_status": "executed",
        "not_promoted_evidence": True,
        "replace_after_execution": False,
        "release_version": project_version,
        "quick_mode": True,
    }
    for key, value in expected_summary_values.items():
        if sample_summary.get(key) != value:
            sample_details.append(f"comprehensive_sample_summary.{key}={sample_summary.get(key)}")
    repeatability = sample_summary.get("repeatability_check", {})
    if repeatability.get("independent_runs", 0) < 2:
        sample_details.append("comprehensive sample requires at least two independent runs")
    if repeatability.get("deterministic_outputs_identical") is not True:
        sample_details.append("comprehensive sample deterministic outputs are not identical")
    if not sample_summary.get("deterministic_output_hashes"):
        sample_details.append("comprehensive sample deterministic hashes missing")
    checks.append(_check("sample input/output executed", "software", not sample_details, sample_details))

    encoding_hits = _mojibake_hits(root)
    checks.append(_check("encoding", "repository", not encoding_hits, encoding_hits[:20]))

    scripts = (
        _load_pyproject(version_root / "pyproject.toml")
        .get("project", {})
        .get("scripts", {})
    )
    expected_scripts = {
        "hidden-attractors": "hidden_attractors.cli.main:main",
    }
    checks.append(
        _check(
            "single public entry point",
            "software",
            scripts == expected_scripts,
            [] if scripts == expected_scripts else [str(scripts)],
        )
    )

    checks.extend(_pypi_readiness_checks(root, version_root, manifest))

    project = _load_pyproject(version_root / "pyproject.toml").get("project", {})
    project_version = str(project.get("version", ""))
    citation_path = root / "CITATION.cff"
    citation_text = citation_path.read_text(encoding="utf-8-sig") if citation_path.exists() else ""
    citation_match = re.search(r'(?m)^version:\s*["\']?([^"\'\s]+)', citation_text)
    manual_text = (version_root / "docs" / "manual_manifest.yaml").read_text(encoding="utf-8-sig")
    manual_match = re.search(r'(?m)^package_version:\s*["\']?([^"\'\s]+)', manual_text)
    zenodo = _load_json(root / ".zenodo.json")
    codemeta = _load_json(root / "codemeta.json")
    version_values = {
        "pyproject": project_version,
        "archive_manifest": str(manifest.get("version", "")),
        "CITATION.cff": citation_match.group(1) if citation_match else "",
        ".zenodo.json": str(zenodo.get("version", "")),
        "codemeta.json": str(codemeta.get("version", "")),
        "manual_manifest": manual_match.group(1) if manual_match else "",
    }
    version_details = [f"{name}={value}" for name, value in version_values.items() if value != project_version]
    checks.append(_check("release version consistency", "software", not version_details, version_details))

    doi_ok = "10.17605/OSF.IO/ZGK74" in citation_text and manifest.get("doi") == "10.17605/OSF.IO/ZGK74"
    checks.append(_check("DOI metadata", "software", doi_ok, [] if doi_ok else ["10.17605/OSF.IO/ZGK74 missing from citation or manifest"]))

    tag_policy_details = []
    if manifest.get("release_tag") != f"v{project_version}":
        tag_policy_details.append(f"release_tag={manifest.get('release_tag')}")
    if manifest.get("source_commit_policy") != "release_tag_resolves_to_source_commit":
        tag_policy_details.append(f"source_commit_policy={manifest.get('source_commit_policy')}")
    for stale_key in ("commit", "commit_status", "last_recorded_freeze_audit_commit"):
        if stale_key in manifest:
            tag_policy_details.append(f"stale self-referential field present: {stale_key}")
    checks.append(_check("release tag source policy", "software", not tag_policy_details, tag_policy_details))

    distribution = manifest.get("scientific_validation", {})
    source_archive_paths = set(manifest.get("source_archive_included_paths", []))
    wheel_paths = set(manifest.get("wheel_distribution_included", []))
    sdist_paths = set(manifest.get("sdist_distribution_included", []))
    archive_only_paths = {
        "version_2/validation/",
        "version_2/release_package/",
    }
    distribution_ok = (
        distribution.get("repository_path") == "version_2/validation/"
        and distribution.get("pypi_distribution") == "excluded"
        and archive_only_paths <= source_archive_paths
        and archive_only_paths.isdisjoint(wheel_paths)
        and archive_only_paths.isdisjoint(sdist_paths)
    )
    checks.append(_check(
        "scientific archive distribution boundary",
        "software",
        distribution_ok,
        [] if distribution_ok else [
            str(distribution),
            f"source_archive_included_paths={sorted(source_archive_paths)}",
            f"wheel_distribution_included={sorted(wheel_paths)}",
            f"sdist_distribution_included={sorted(sdist_paths)}",
        ],
    ))

    public_release_files = [
        release_root / "README_RELEASE.md",
        release_root / "PROGRAM_SUMMARY.md",
        release_root / "PUBLISHING_POLICY.md",
        release_root / "SAMPLE_RUN.md",
        release_root / "archive_manifest.json",
    ]
    blocked_release_patterns = (
        re.compile(r"\bplanned\s+support\b"),
        re.compile(r"\bfuture\s+work\b"),
        re.compile(r"\binternal\s+research\s+plan\b"),
        re.compile(r"\bcandidate-specific\s+promotion\s+boundary\b"),
    )
    narrative_details = []
    for path in public_release_files:
        text = path.read_text(encoding="utf-8-sig").lower()
        for pattern in blocked_release_patterns:
            if pattern.search(text):
                narrative_details.append(
                    f"{path.relative_to(root).as_posix()}: blocked forward-looking phrase"
                )
    checks.append(_check("public release narrative", "software", not narrative_details, narrative_details))

    verification = manifest.get("release_verification", {})
    verification_ok = (
        verification.get("workflow") == ".github/workflows/publish-pypi.yml"
        and verification.get("publication_gate") == "protected GitHub environment named pypi"
        and verification.get("authentication") == "PyPI Trusted Publishing with job-scoped OIDC"
    )
    checks.append(_check(
        "release verification policy",
        "software",
        verification_ok,
        [] if verification_ok else [str(verification)],
    ))

    release_pair = (
        manifest.get("release_state"),
        manifest.get("publication_status"),
    )
    readiness_ok = (
        release_pair in PUBLICATION_STATE_PAIRS
        and manifest.get("repository_readiness") == "ready"
        and manifest.get("software_package_readiness") == "ready"
        and manifest.get("release_preparation_readiness") == "ready"
    )
    checks.append(_check("readiness levels", "software", readiness_ok, [] if readiness_ok else [
        f"release_state={manifest.get('release_state')}",
        f"publication_status={manifest.get('publication_status')}",
        f"repository_readiness={manifest.get('repository_readiness')}",
        f"software_package_readiness={manifest.get('software_package_readiness')}",
        f"release_preparation_readiness={manifest.get('release_preparation_readiness')}",
    ]))

    failures = [c for c in checks if not c["ok"]]
    repository_readiness = "failed" if any(c["category"] == "repository" and not c["ok"] for c in checks) else "passed"
    software_package_readiness = "failed" if any(c["category"] == "software" and not c["ok"] for c in checks) else "passed"
    release_preparation_readiness = "failed" if failures else "passed"
    status = "failed" if failures else "passed"

    payload = {
        "status": status,
        "strict": bool(args.strict or args.submission_strict),
        "target_version": project_version,
        "release_state": manifest.get("release_state"),
        "publication_status": manifest.get("publication_status"),
        "repository_readiness": repository_readiness,
        "software_package_readiness": software_package_readiness,
        "release_preparation_readiness": release_preparation_readiness,
        "checks": checks,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Release readiness: {status}")
        print(f"repository_readiness: {repository_readiness}")
        print(f"software_package_readiness: {software_package_readiness}")
        print(f"release_preparation_readiness: {release_preparation_readiness}")
        print(f"publication_status: {manifest.get('publication_status')}")
        for check in checks:
            label = "ok" if check["ok"] else "fail"
            print(f"- {label}: {check['name']}")
            for item in check["details"]:
                print(f"  - {item}")
    exit_code = 1 if failures else 0
    sys.exit(exit_code)



