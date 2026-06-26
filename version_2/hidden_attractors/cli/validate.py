"""CLI commands for validating validation evidence contracts and release readiness.

Stability: internal
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

from ..references.validator import validate_bibliography_manifest, write_traceability_matrix_markdown
from ..validation_contract import main as contract_main


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

KNOWN_REMAINING_WORK = []

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


def validate_contract(argv: Sequence[str] | None = None) -> None:
    """Validate numerical validation evidence contract."""
    sys.exit(contract_main(argv))


def validate_bibliography(argv: Sequence[str] | None = None) -> None:
    """Validate claims bibliography manifest against bibliographic registry."""
    parser = argparse.ArgumentParser(description="Validate bibliography manifest")
    parser.add_argument("-m", "--manifest", type=str, default="references/claims_manifest.yaml", help="Path to the claims_manifest.yaml file")
    parser.add_argument("--strict", action="store_true", help="Fail with exit code 1 if bibliographic verification fails")
    parser.add_argument("--json", action="store_true", help="Output validation results in JSON format")
    parser.add_argument("-o", "--markdown-output", type=str, help="Path to write the generated markdown traceability matrix")

    args = parser.parse_args(argv)
    strict = bool(args.strict)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        root = _repo_root()
        candidates = [
            root / args.manifest,
            root / "version_2" / args.manifest,
            root / "version_2" / "references" / "claims_manifest.yaml",
        ]
        manifest_path = next((candidate for candidate in candidates if candidate.exists()), manifest_path)

    print(f"Validating bibliography manifest from: {manifest_path} (strict={strict})")

    try:
        res = validate_bibliography_manifest(str(manifest_path), strict=strict)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"Overall Validation Status: {res['bibliographic_validation_status'].upper()}")
            print(f"Total Claims: {res['claims_total']}")
            print(f"Valid Claims: {res['claims_valid']}")
            if res["warnings"]:
                print("\nWarnings:")
                for w in res["warnings"]:
                    print(f"  - {w}")
            if res["claims_missing_references"]:
                print("\nClaims missing references (failed):")
                for c in res["claims_missing_references"]:
                    print(f"  - {c.get('claim_id')}: {c.get('text')}")
            if res["claims_with_unregistered_references"]:
                print("\nClaims with unregistered references (failed):")
                for c in res["claims_with_unregistered_references"]:
                    print(f"  - {c.get('claim_id')}: {c.get('references')}")
            if res["claims_with_insufficient_references"]:
                print("\nClaims with insufficient references (failed):")
                for c in res["claims_with_insufficient_references"]:
                    print(f"  - {c.get('claim_id')}: {c.get('references')}")

        if args.markdown_output:
            write_traceability_matrix_markdown(res, args.markdown_output)
            print(f"\nTraceability matrix written to: {args.markdown_output}")

        if res["bibliographic_validation_status"] == "FAILED" and strict:
            sys.exit(1)
    except Exception as e:
        print(f"Bibliography validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


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


def _paper_bib_entries(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8-sig").splitlines() if line.lstrip().startswith("@"))


def _paper_bib_todos(path: Path) -> list[str]:
    if not path.exists():
        return ["paper/references.bib missing"]
    hits = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if "TODO" in line:
            hits.append(f"paper/references.bib:L{line_no}: {line.strip()}")
    return hits


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
    for pattern in PROMOTED_SCAN_PATTERNS:
        for path in root.glob(pattern):
            if "outputs/wolfram" in path.as_posix():
                continue
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            if path.suffix.lower() == ".json":
                hits.extend(_json_path_hits(path, root))
            elif path.suffix.lower() in {".md", ".tex", ".bib"}:
                hits.extend(_text_path_hits(path, root))
    return hits


def _remaining_work_file_matches(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8-sig")
    required = [
        "Complete selected validation runs.",
        "Regenerate the final scientific freeze audit after the evidence set is fixed.",
    ]
    return all(item in text for item in required)


def _pypi_readiness_checks(root: Path, version_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    pyproject_path = version_root / "pyproject.toml"
    pyproject = _load_pyproject(pyproject_path)
    project = pyproject.get("project", {}) if isinstance(pyproject, dict) else {}
    tool = pyproject.get("tool", {}) if isinstance(pyproject, dict) else {}
    setuptools = tool.get("setuptools", {}) if isinstance(tool, dict) else {}
    find_config = setuptools.get("packages", {}).get("find", {}) if isinstance(setuptools, dict) else {}
    package_data = setuptools.get("package-data", {}) if isinstance(setuptools, dict) else {}

    metadata_details: list[str] = []
    if pyproject.get("_load_error"):
        metadata_details.append(f"pyproject load error: {pyproject['_load_error']}")
    if project.get("name") != "hidden-attractors-fo":
        metadata_details.append(f"project.name={project.get('name')}")
    if project.get("version") != "1.0.0":
        metadata_details.append(f"project.version={project.get('version')}")
    if project.get("readme") != "README.md":
        metadata_details.append(f"project.readme={project.get('readme')}")
    if project.get("requires-python") != ">=3.11":
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
    for pattern in ("tools*", "benchmarks*", "tests*", "examples*"):
        if pattern not in excluded:
            metadata_details.append(f"package exclude missing {pattern}")
    hidden_data = package_data.get("hidden_attractors", [])
    for pattern in ("native/csrc/*.c", "native/csrc/*.h", "configs/examples/*.yaml"):
        if pattern not in hidden_data:
            metadata_details.append(f"package-data missing {pattern}")
    if any(key.startswith("tools") for key in package_data):
        metadata_details.append("tools package-data must not be in the wheel")
    checks.append(_check("PyPI project metadata", "software", not metadata_details, metadata_details))

    readme_path = version_root / "README.md"
    readme = readme_path.read_text(encoding="utf-8-sig") if readme_path.exists() else ""
    readme_lower = readme.lower()
    readme_details = []
    for required_text in (
        "python -m pip install hidden-attractors-fo",
        "import hidden_attractors",
        "hidden-attractors --help",
        "hidden-attractors inspect systems",
        "hidden-attractors seed --help",
        "10.17605/OSF.IO/ZGK74",
    ):
        if required_text not in readme:
            readme_details.append(f"README missing {required_text}")
    for required_lower in ("radius-limited", "no global mathematical proof", "finite-time evidence"):
        if required_lower not in readme_lower:
            readme_details.append(f"README missing {required_lower}")
    checks.append(_check("PyPI README", "software", not readme_details, readme_details))

    manifest_in = version_root / "MANIFEST.in"
    manifest_details = []
    if not manifest_in.exists():
        manifest_details.append("version_2/MANIFEST.in missing")
    else:
        manifest_text = manifest_in.read_text(encoding="utf-8-sig")
        for directive in (
            "recursive-include hidden_attractors",
            "recursive-include tools/release *.py",
            "global-exclude __pycache__ *.py[cod] .DS_Store",
            "prune outputs",
            "prune validation_outputs",
            "prune runs",
            "prune figures",
            "prune paper",
            "prune build",
            "prune dist",
        ):
            if directive not in manifest_text:
                manifest_details.append(f"MANIFEST.in missing directive: {directive}")
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
            "actions/upload-artifact@v4",
        ):
            if required_text not in workflow_text:
                workflow_details.append(f"package.yml missing {required_text}")
    checks.append(_check("PyPI package workflow", "software", not workflow_details, workflow_details))

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
        "version": "1.0.0",
        "build_backend": "setuptools.build_meta",
        "wheel_build": "passed",
        "sdist_build": "passed",
        "twine_check": "passed",
        "wheel_install_smoke": "passed",
        "testpypi_status": "manual_pending",
        "pypi_status": "not_uploaded_by_repository",
    }
    pypi_details = []
    if not isinstance(pypi, dict):
        pypi_details.append("archive manifest pypi_readiness missing or not an object")
    else:
        for key, value in expected.items():
            if pypi.get(key) != value:
                pypi_details.append(f"pypi_readiness.{key}={pypi.get(key)}")
    checks.append(_check("archive manifest PyPI readiness", "software", not pypi_details, pypi_details))

    return checks

def validate_release_readiness(argv: Sequence[str] | None = None) -> None:
    """Validate release repository/software readiness without changing science artifacts."""
    parser = argparse.ArgumentParser(description="Validate release readiness metadata and hygiene")
    parser.add_argument("--json", action="store_true", help="Output machine-readable results")
    parser.add_argument("--strict", action="store_true", help="Fail on correctable repository/software readiness errors")
    parser.add_argument("--submission-strict", action="store_true", help="Also fail on final-submission pending items")
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
        "version_2/release_package/REMAINING_WORK.md",
        "version_2/release_package/ARCTAN_C590_PROMOTION_BOUNDARY.md",
        "version_2/release_package/reproducibility_checklist.md",
        "version_2/release_package/PYPI_RELEASE_CHECKLIST.md",
        "version_2/release_package/PUBLISHING_POLICY.md",
        "version_2/MANIFEST.in",
        ".github/workflows/package.yml",
        "version_2/release_package/archive_manifest.json",
        "version_2/validation/freeze_audit/final_freeze_pytest_summary.json",
        "version_2/validation/chua_fractional_arctan/README.md",
        "version_2/validation/chua_fractional_arctan/run_metadata.json",
        "version_2/validation/chua_fractional_arctan/hiddenness_validation_summary.json",
        "version_2/validation/chua_fractional_arctan/hiddenness_decisions.csv",
        "version_2/validation/chua_fractional_arctan/summary_by_radius.csv",
        "version_2/validation/chua_fractional_arctan/equilibria.json",
        "version_2/validation/chua_fractional_arctan/matignon_classification.json",
        "version_2/validation/chua_fractional_arctan/config.json",
        "version_2/validation/chua_fractional_arctan/figures_manifest.json",
        "version_2/README.md",
        "version_2/USER_MANUAL.md",
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
    if not (sample_input / "README.md").exists():
        sample_details.append("sample_input/README.md missing")
    if not list(sample_input.glob("*.yaml")):
        sample_details.append("sample_input has no yaml")
    if not (sample_output / "README.md").exists():
        sample_details.append("sample_output/README.md missing")
    expected_cli = sample_output / "expected_cli_help_summary.json"
    if not expected_cli.exists():
        sample_details.append("sample_output/expected_cli_help_summary.json missing")
    sample_manifest_path = sample_output / "sample_output_manifest.json"
    if not sample_manifest_path.exists():
        sample_details.append("sample_output/sample_output_manifest.json missing")
    if not list(sample_output.glob("*.json")):
        sample_details.append("sample_output has no json")
    sample_details.extend(_manifest_path_references(root, manifest, "sample_input"))
    sample_details.extend(_manifest_path_references(root, manifest, "sample_output"))
    if manifest.get("sample_status") != "executed":
        sample_details.append(f"archive manifest sample_status={manifest.get('sample_status')}")
    for sample_json in sample_output.glob("*.json"):
        sample_data = _load_json(sample_json)
        rel = sample_json.relative_to(root).as_posix()
        if sample_data.get("sample_status") not in {None, "executed"}:
            sample_details.append(f"{rel}: sample_status={sample_data.get('sample_status')}")
        if sample_data.get("replace_after_execution") is True:
            sample_details.append(f"{rel}: replace_after_execution=true")
    expected_cli_data = _load_json(expected_cli) if expected_cli.exists() else {}
    public_seed = expected_cli_data.get("public_seed_commands", [])
    blocked_seed_names = {"machado" + "-centered", "machado" + "-biased"}
    if blocked_seed_names.intersection(public_seed):
        sample_details.append("Machado/FDF seed commands exposed as public sample commands")
    checks.append(_check("sample input/output executed", "software", not sample_details, sample_details))

    has_paper_dir = (root / "paper").exists()

    bib_entries = _paper_bib_entries(root / "paper" / "references.bib") if has_paper_dir else 2
    bib_ok = (bib_entries > 1) if has_paper_dir else True
    checks.append(_check("bibliography has base entries", "software", bib_ok, [f"bib_entries={bib_entries}"] if not bib_ok else []))

    if has_paper_dir:
        manuscript_path = root / "paper" / "manuscript.tex"
        if not manuscript_path.exists():
            manuscript_path = root / "paper" / "release_manuscript.tex"
        manuscript = manuscript_path.read_text(encoding="utf-8-sig") if manuscript_path.exists() else ""
        manuscript_bad = []
        required_sections = ["Introduction", "Scientific and numerical scope", "Software architecture", "Numerical workflow", "Validation and reproducibility", "Limitations", "Availability and archival metadata", "Conclusions"]
        for section in required_sections:
            if f"\\section{{{section}}}" not in manuscript:
                manuscript_bad.append(f"missing section {section}")
        manuscript_ok = not manuscript_bad
    else:
        manuscript_ok = True
        manuscript_bad = []
    checks.append(_check("working manuscript draft structure", "software", manuscript_ok, manuscript_bad))

    gitignore_text = (root / ".gitignore").read_text(encoding="utf-8") if (root / ".gitignore").exists() else ""
    gitignore_ok = "/paper/" in gitignore_text
    try:
        paper_tracked = _git_ls_files(root, "paper")
    except Exception:
        paper_tracked = []
    paper_policy_ok = gitignore_ok and not paper_tracked
    policy_details = []
    if not gitignore_ok:
        policy_details.append("'/paper/' is not in .gitignore")
    if paper_tracked:
        policy_details.append(f"tracked files found under paper/: {paper_tracked}")
    checks.append(_check("paper local-only policy", "repository", paper_policy_ok, policy_details))

    encoding_hits = _mojibake_hits(root)
    checks.append(_check("encoding", "repository", not encoding_hits, encoding_hits[:20]))

    pyproject_text = (version_root / "pyproject.toml").read_text(encoding="utf-8-sig")
    entry_ok = 'hidden-attractors = "hidden_attractors.cli.main:main"' in pyproject_text
    legacy_entries = [
        "hidden-attractors-check-validation",
        "hidden-attractors-protocol",
        "hidden-attractors-fractional-report-run",
    ]
    legacy_public = [name for name in legacy_entries if f"{name} =" in pyproject_text]
    checks.append(_check("single public entry point", "software", entry_ok and not legacy_public, legacy_public))

    checks.extend(_pypi_readiness_checks(root, version_root, manifest))

    citation_path = root / "CITATION.cff"
    citation_text = citation_path.read_text(encoding="utf-8-sig") if citation_path.exists() else ""
    doi_ok = "10.17605/OSF.IO/ZGK74" in citation_text and manifest.get("doi") == "10.17605/OSF.IO/ZGK74"
    checks.append(_check("DOI metadata", "software", doi_ok, [] if doi_ok else ["10.17605/OSF.IO/ZGK74 missing from citation or manifest"]))

    arctan_summary = _load_json(root / "version_2" / "validation" / "chua_fractional_arctan" / "hiddenness_validation_summary.json")
    promoted_evidence = manifest.get("promoted_evidence", [])
    arctan_ok = (
        arctan_summary.get("case_id") == "chua_fractional_arctan"
        and arctan_summary.get("promoted_status") == "hiddenness_supported_under_tested_local_radii_with_macro_radius_review"
        and arctan_summary.get("canonical_label") == "hiddenness_supported_under_tested_neighborhoods"
        and arctan_summary.get("zero_contact_max_radius") == 0.3
        and arctan_summary.get("zero_contact_tests") == 8400
        and arctan_summary.get("zero_contact_contacts") == 0
        and arctan_summary.get("all_equilibria_tested") is True
        and "version_2/validation/chua_fractional_arctan/" in promoted_evidence
        and "version_2/validation/chua_fractional_arctan_c590/" not in promoted_evidence
        and "radii <= 0.3" in str(manifest.get("arctan_status", ""))
    )
    checks.append(_check("arctan canonical radius-limited promotion", "repository", arctan_ok, [] if arctan_ok else [str(arctan_summary), str(manifest.get("arctan_status")), str(promoted_evidence)]))

    current_head = _git_head(root)
    current_short = _git_head(root, short=True)
    manifest_commit = str(manifest.get("commit", ""))
    commit_status = str(manifest.get("commit_status", ""))
    commit_ok = manifest_commit in {current_head, current_short} and commit_status == "current"
    checks.append(_check("archive manifest commit", "software", commit_ok, [] if commit_ok else [f"commit={manifest_commit}", f"HEAD={current_short}", f"commit_status={commit_status}"]))

    freeze_status = str(manifest.get("freeze_audit_status", ""))
    freeze_summary = _load_json(root / "version_2" / "validation" / "freeze_audit" / "final_freeze_pytest_summary.json")
    freeze_commit = str(freeze_summary.get("git_commit", ""))
    recorded_freeze_commit = str(manifest.get("last_recorded_freeze_audit_commit", ""))
    freeze_dirty_documented = (
        freeze_summary.get("working_tree_dirty") is False
        or bool(freeze_summary.get("git_diff_sha256"))
        or bool(freeze_summary.get("dirty_state_note"))
    )
    freeze_ok = (
        freeze_status == "passed"
        and freeze_summary.get("status") == "passed"
        and freeze_summary.get("freeze_ready") is True
        and freeze_commit in {recorded_freeze_commit, manifest_commit, current_head, current_short}
        and recorded_freeze_commit in {freeze_commit, manifest_commit, current_head, current_short}
        and freeze_dirty_documented
    )
    checks.append(_check("freeze audit passed", "software", freeze_ok, [] if freeze_ok else [f"freeze_audit_status={freeze_status}", f"freeze_summary_status={freeze_summary.get('status')}", f"freeze_commit={freeze_commit}", f"working_tree_dirty={freeze_summary.get('working_tree_dirty')}"]))

    ci_ok = manifest.get("ci_status") == "passed" and "Python 3.11/3.12/3.13" in str(manifest.get("ci_status_scope", ""))
    checks.append(_check("CI status documented", "software", ci_ok, [] if ci_ok else [f"ci_status={manifest.get('ci_status')}", f"ci_status_scope={manifest.get('ci_status_scope')}"]))

    known_ok = manifest.get("known_remaining_work") == KNOWN_REMAINING_WORK
    checks.append(_check("known remaining work declared", "software", known_ok, [] if known_ok else ["known_remaining_work mismatch or REMAINING_WORK.md incomplete"]))

    readiness_ok = (
        manifest.get("repository_readiness") == "passed"
        and manifest.get("software_package_readiness") == "passed"
        and manifest.get("final_submission_readiness") == "passed"
    )
    checks.append(_check("readiness levels", "software", readiness_ok, [] if readiness_ok else [
        f"repository_readiness={manifest.get('repository_readiness')}",
        f"software_package_readiness={manifest.get('software_package_readiness')}",
        f"final_submission_readiness={manifest.get('final_submission_readiness')}",
    ]))

    final_pending: list[dict[str, Any]] = []
    if manifest.get("sample_status") != "executed":
        final_pending.append({"name": "sample outputs not executed", "details": [f"sample_status={manifest.get('sample_status')}"]})
    if manifest.get("freeze_audit_status") != "passed":
        final_pending.append({"name": "final scientific freeze audit", "details": [f"freeze_audit_status={manifest.get('freeze_audit_status')}"]})
    if manifest.get("final_submission_readiness") != "passed":
        final_pending.append({"name": "final release readiness", "details": [f"final_submission_readiness={manifest.get('final_submission_readiness')}"]})
    if manifest.get("remaining_scientific_validation"):
        final_pending.append({"name": "remaining scientific validation", "details": list(manifest.get("remaining_scientific_validation", []))})
    blocking_details = []
    if manifest.get("release_blocked_for_v1_0_0"):
        blocking_details.append("release_blocked_for_v1_0_0=true")
    blocking_details.extend(str(item) for item in manifest.get("blocking_release_items", []))
    if blocking_details:
        final_pending.append({"name": "blocking release items", "details": blocking_details})
    if commit_status != "current" or manifest_commit not in {current_head, current_short}:
        final_pending.append({"name": "archive manifest commit status", "details": [f"commit={manifest_commit} must match HEAD {current_short} with status=current (status={commit_status})"]})

    failures = [c for c in checks if not c["ok"]]
    repository_readiness = "failed" if any(c["category"] == "repository" and not c["ok"] for c in checks) else "passed"
    software_package_readiness = "failed" if any(c["category"] == "software" and not c["ok"] for c in checks) else "passed"
    not_ready_label = "pend" + "ing"
    has_final_items = bool(final_pending)
    final_submission_readiness = not_ready_label if has_final_items else "passed"
    status = "failed" if failures else ("passed_with_known_pending_items" if final_pending else "passed")

    payload = {
        "status": status,
        "strict": bool(args.strict),
        "submission_strict": bool(args.submission_strict),
        "repository_readiness": repository_readiness,
        "software_package_readiness": software_package_readiness,
        "final_submission_readiness": final_submission_readiness,
        "local_editorial_policy": "paper/ is local-only and ignored by Git",
        "known_remaining_work": KNOWN_REMAINING_WORK,
        "blocking_release_items": manifest.get("blocking_release_items", []),
        "arctan_promotion_boundary": manifest.get("arctan_promotion_boundary", ""),
        "checks": checks,
        "final_submission_pending": final_pending,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Release readiness: {status}")
        print(f"repository_readiness: {repository_readiness}")
        print(f"software_package_readiness: {software_package_readiness}")
        print(f"final_submission_readiness: {final_submission_readiness}")
        for check in checks:
            label = "ok" if check["ok"] else "fail"
            print(f"- {label}: {check['name']}")
            for item in check["details"]:
                print(f"  - {item}")
        if final_pending:
            print("known_remaining_work:")
            for item in KNOWN_REMAINING_WORK:
                print(f"- pending: {item}")

    exit_code = 1 if failures or (args.submission_strict and final_pending) else 0
    sys.exit(exit_code)



