"""CLI commands for validating validation evidence contracts and release readiness.

Stability: internal
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
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
    parser.add_argument("-m", "--manifest", type=str, default="version_2/references/claims_manifest.yaml", help="Path to the claims_manifest.yaml file")
    parser.add_argument("--strict", action="store_true", help="Fail with exit code 1 if bibliographic verification fails")
    parser.add_argument("--json", action="store_true", help="Output validation results in JSON format")
    parser.add_argument("-o", "--markdown-output", type=str, help="Path to write the generated markdown traceability matrix")

    args = parser.parse_args(argv)
    strict = bool(args.strict)
    manifest_path = args.manifest

    print(f"Validating bibliography manifest from: {manifest_path} (strict={strict})")

    try:
        res = validate_bibliography_manifest(manifest_path, strict=strict)
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
        "version_2/release_package/archive_manifest.json",
        "version_2/validation/freeze_audit/final_freeze_pytest_summary.json",
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
    if not list(sample_output.glob("*.json")):
        sample_details.append("sample_output has no json")
    sample_details.extend(_manifest_path_references(root, manifest, "sample_input"))
    sample_details.extend(_manifest_path_references(root, manifest, "sample_output"))
    checks.append(_check("sample input/output templates", "software", not sample_details, sample_details))

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

    citation_path = root / "CITATION.cff"
    citation_text = citation_path.read_text(encoding="utf-8-sig") if citation_path.exists() else ""
    doi_ok = "10.17605/OSF.IO/ZGK74" in citation_text and manifest.get("doi") == "10.17605/OSF.IO/ZGK74"
    checks.append(_check("DOI metadata", "software", doi_ok, [] if doi_ok else ["10.17605/OSF.IO/ZGK74 missing from citation or manifest"]))

    arctan_summary = _load_json(root / "version_2" / "validation" / "chua_fractional_arctan_c590" / "validation_summary.json")
    arctan_ok = (
        arctan_summary.get("promoted_status") == "hiddenness_supported_under_tested_local_radii_with_macro_radius_review"
        and arctan_summary.get("zero_contact_max_radius") == 0.3
        and arctan_summary.get("zero_contact_tests") == 8400
        and arctan_summary.get("zero_contact_contacts") == 0
        and "radii <= 0.3" in str(manifest.get("arctan_status", ""))
    )
    checks.append(_check("arctan c590 radius-limited promotion", "repository", arctan_ok, [] if arctan_ok else [str(arctan_summary), str(manifest.get("arctan_status"))]))

    current_head = _git_head(root)
    current_short = _git_head(root, short=True)
    manifest_commit = str(manifest.get("commit", ""))
    commit_status = str(manifest.get("commit_status", ""))
    commit_ok = (
        manifest_commit in {current_head, current_short}
        and commit_status == "current"
    ) or commit_status == "pending_update_after_final_cleanup_commit"
    checks.append(_check("archive manifest commit", "software", commit_ok, [] if commit_ok else [f"commit={manifest_commit}", f"HEAD={current_short}", f"commit_status={commit_status}"]))

    freeze_status = str(manifest.get("freeze_audit_status", ""))
    freeze_ok = freeze_status == "pending_final_scientific_freeze" and str(manifest.get("last_recorded_freeze_audit_commit", "")).startswith("2bcea343")
    checks.append(_check("freeze audit separated from CI", "software", freeze_ok, [] if freeze_ok else [f"freeze_audit_status={freeze_status}"]))

    ci_ok = manifest.get("ci_status") == "passed" and "Python 3.11/3.12/3.13" in str(manifest.get("ci_status_scope", ""))
    checks.append(_check("CI status documented", "software", ci_ok, [] if ci_ok else [f"ci_status={manifest.get('ci_status')}", f"ci_status_scope={manifest.get('ci_status_scope')}"]))

    known_ok = manifest.get("known_remaining_work") == KNOWN_REMAINING_WORK
    checks.append(_check("known remaining work declared", "software", known_ok, [] if known_ok else ["known_remaining_work mismatch or REMAINING_WORK.md incomplete"]))

    readiness_ok = (
        manifest.get("repository_readiness") == "passed"
        and manifest.get("software_package_readiness") == "passed"
        and manifest.get("final_submission_readiness") in {"pending", "passed"}
    )
    checks.append(_check("readiness levels", "software", readiness_ok, [] if readiness_ok else [
        f"repository_readiness={manifest.get('repository_readiness')}",
        f"software_package_readiness={manifest.get('software_package_readiness')}",
        f"final_submission_readiness={manifest.get('final_submission_readiness')}",
    ]))

    final_pending: list[dict[str, Any]] = []
    bib_todos = _paper_bib_todos(root / "paper" / "references.bib")
    if bib_todos:
        final_pending.append({"name": "bibliographic metadata TODOs", "details": bib_todos[:20]})
    if manifest.get("sample_status") == "template_only_pending_execution":
        final_pending.append({"name": "sample outputs not executed", "details": ["sample_status=template_only_pending_execution"]})
    if manifest.get("freeze_audit_status") == "pending_final_scientific_freeze":
        final_pending.append({"name": "final scientific freeze audit", "details": ["freeze audit must be regenerated after final scientific evidence is frozen"]})
    if manifest.get("final_submission_readiness") == "pending":
        final_pending.append({"name": "final release manuscript/template work", "details": ["final manuscript remains editorial work"]})
    if manifest.get("remaining_scientific_validation"):
        final_pending.append({"name": "remaining scientific validation", "details": list(manifest.get("remaining_scientific_validation", []))})
    blocking_details = []
    if manifest.get("release_blocked_for_v1_0_0"):
        blocking_details.append("release_blocked_for_v1_0_0=true")
    blocking_details.extend(str(item) for item in manifest.get("blocking_release_items", []))
    if blocking_details:
        final_pending.append({"name": "blocking release items", "details": blocking_details})
    if manifest.get("commit_status") == "pending_update_after_final_cleanup_commit" or manifest_commit not in {current_head, current_short}:
        final_pending.append({"name": "archive manifest commit status", "details": [f"commit={manifest_commit} is pending update to final commit hash (status={commit_status})"]})

    failures = [c for c in checks if not c["ok"]]
    repository_readiness = "failed" if any(c["category"] == "repository" and not c["ok"] for c in checks) else "passed"
    software_package_readiness = "failed" if any(c["category"] == "software" and not c["ok"] for c in checks) else "passed"
    final_submission_readiness = "pending" if final_pending else "passed"
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
