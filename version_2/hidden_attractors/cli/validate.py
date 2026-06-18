"""CLI commands for validating validation evidence contracts and claims bibliography.

Stability: internal
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from ..references.validator import validate_bibliography_manifest, write_traceability_matrix_markdown
from ..validation_contract import main as contract_main


MOJIBAKE_PATTERNS = [
    "Ãƒ",
    "Ã‚",
    "Ã¢â€\x9dâ‚¬",
    "Ã¢â‚¬â€œ",
    "Ã¢â‚¬â„¢",
    "Ă",
    "â€œ",
    "â€",
    "â”",
]

MAIN_TEXT_PATTERNS = [
    "README.md",
    "CHANGELOG.md",
    "RELEASE_NOTES.md",
    "REPRODUCIBILITY.md",
    "CITATION.cff",
    ".zenodo.json",
    "codemeta.json",
    "paper/*.tex",
    "paper/*.md",
    "paper/*.bib",
    "version_2/README.md",
    "version_2/USER_MANUAL.md",
    "version_2/MANIFEST.md",
    "version_2/pyproject.toml",
    "version_2/docs/*.md",
    "version_2/cpc_submission/*.md",
    "version_2/cpc_submission/*.json",
    "version_2/cpc_submission/sample_input/*.yaml",
    "version_2/cpc_submission/sample_input/*.md",
    "version_2/cpc_submission/sample_output/*.json",
    "version_2/cpc_submission/sample_output/*.md",
]


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

        if res["bibliographic_validation_status"] == "failed" and strict:
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


def _check(name: str, level: str, ok: bool, details: list[str] | None = None) -> dict[str, object]:
    return {"name": name, "level": level, "ok": ok, "details": details or []}


def _load_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def _paper_bib_entries(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.lstrip().startswith("@"))


def _mojibake_hits(root: Path) -> list[str]:
    hits: list[str] = []
    seen: set[Path] = set()
    for pattern in MAIN_TEXT_PATTERNS:
        for path in root.glob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                hits.append(f"{path.relative_to(root).as_posix()}: utf-8 decode failed: {exc}")
                continue
            for bad in MOJIBAKE_PATTERNS:
                if bad in text:
                    hits.append(f"{path.relative_to(root).as_posix()}: contains {bad!r}")
                    break
    return hits


def _manifest_path_references(root: Path, manifest: dict[str, object], key: str) -> list[str]:
    values = manifest.get(key, [])
    if not isinstance(values, list):
        return [f"{key} is not a list"]
    missing = []
    for rel in values:
        if not isinstance(rel, str) or not (root / rel).exists():
            missing.append(str(rel))
    return missing


def validate_cpc_readiness(argv: Sequence[str] | None = None) -> None:
    """Validate CPC metadata and repository hygiene without changing science artifacts."""
    parser = argparse.ArgumentParser(description="Validate CPC readiness metadata and hygiene")
    parser.add_argument("--json", action="store_true", help="Output machine-readable results")
    parser.add_argument("--strict", action="store_true", help="Treat CPC-finalization warnings as failures")
    args = parser.parse_args(argv)

    root = _repo_root()
    version_root = root / "version_2"
    manifest_path = version_root / "cpc_submission" / "archive_manifest.json"
    manifest = _load_json(manifest_path)
    checks: list[dict[str, object]] = []

    required = [
        "CITATION.cff",
        ".zenodo.json",
        "codemeta.json",
        "AUTHORS.md",
        "CHANGELOG.md",
        "RELEASE_NOTES.md",
        "REPRODUCIBILITY.md",
        "paper/cpc_program_summary.tex",
        "paper/cpc_manuscript.tex",
        "paper/references.bib",
        "version_2/cpc_submission/README_CPC.md",
        "version_2/cpc_submission/PROGRAM_SUMMARY.md",
        "version_2/cpc_submission/SAMPLE_RUN.md",
        "version_2/cpc_submission/reproducibility_checklist.md",
        "version_2/cpc_submission/archive_manifest.json",
        "version_2/validation/freeze_audit/final_freeze_pytest_summary.json",
        "version_2/README.md",
        "version_2/USER_MANUAL.md",
    ]
    missing = [rel for rel in required if not (root / rel).exists()]
    checks.append(_check("required metadata", "fail", not missing, missing))

    validation_outputs_tracked = _git_ls_files(root, "version_2/validation_outputs")
    checks.append(_check("validation_outputs untracked", "fail", not validation_outputs_tracked, validation_outputs_tracked))

    sample_input = version_root / "cpc_submission" / "sample_input"
    sample_output = version_root / "cpc_submission" / "sample_output"
    sample_details = []
    if not (sample_input / "README.md").exists():
        sample_details.append("sample_input/README.md missing")
    if not list(sample_input.glob("*.yaml")):
        sample_details.append("sample_input has no yaml")
    if not (sample_output / "README.md").exists():
        sample_details.append("sample_output/README.md missing")
    if not list(sample_output.glob("*.json")):
        sample_details.append("sample_output has no json")
    sample_details.extend(_manifest_path_references(root, manifest, "sample_input"))
    sample_details.extend(_manifest_path_references(root, manifest, "sample_output"))
    checks.append(_check("sample input/output", "fail", not sample_details, sample_details))

    bib_entries = _paper_bib_entries(root / "paper" / "references.bib")
    checks.append(_check("bibliography", "fail", bib_entries > 1, [f"bib_entries={bib_entries}"] if bib_entries <= 1 else []))

    manuscript_path = root / "paper" / "cpc_manuscript.tex"
    manuscript = manuscript_path.read_text(encoding="utf-8") if manuscript_path.exists() else ""
    manuscript_bad = []
    if "placeholder manuscript" in manuscript.lower() or "placeholder" in manuscript.lower():
        manuscript_bad.append("placeholder wording remains")
    required_sections = ["Introduction", "Scientific and numerical scope", "Software architecture", "Numerical workflow", "Validation and reproducibility", "Limitations", "Availability and archival metadata", "Conclusions"]
    for section in required_sections:
        if f"\\section{{{section}}}" not in manuscript:
            manuscript_bad.append(f"missing section {section}")
    checks.append(_check("manuscript", "fail", not manuscript_bad, manuscript_bad))

    encoding_hits = _mojibake_hits(root)
    checks.append(_check("encoding", "fail", not encoding_hits, encoding_hits[:20]))

    pyproject_text = (version_root / "pyproject.toml").read_text(encoding="utf-8")
    entry_ok = 'hidden-attractors = "hidden_attractors.cli.main:main"' in pyproject_text
    legacy_entries = [
        "hidden-attractors-check-validation",
        "hidden-attractors-protocol",
        "hidden-attractors-fractional-report-run",
    ]
    legacy_public = [name for name in legacy_entries if f"{name} =" in pyproject_text]
    checks.append(_check("single public entry point", "fail", entry_ok and not legacy_public, legacy_public))

    citation_path = root / "CITATION.cff"
    citation_text = citation_path.read_text(encoding="utf-8") if citation_path.exists() else ""
    doi_ok = "10.17605/OSF.IO/ZGK74" in citation_text
    checks.append(_check("DOI", "fail", doi_ok, [] if doi_ok else ["10.17605/OSF.IO/ZGK74 missing"]))

    arctan_promoted = _git_ls_files(root, "version_2/validation/chua_fractional_arctan")
    checks.append(_check("arctan not promoted", "fail", not arctan_promoted, arctan_promoted))

    current_head = _git_head(root)
    current_short = _git_head(root, short=True)
    manifest_commit = str(manifest.get("commit", ""))
    commit_status = str(manifest.get("commit_status", ""))
    commit_ok = manifest_commit in {current_head, current_short} or commit_status == "pending_update_after_final_audit"
    commit_warn = [] if manifest_commit in {current_head, current_short} and commit_status == "current" else [f"commit={manifest_commit}", f"HEAD={current_short}", f"commit_status={commit_status}"]
    checks.append(_check("archive manifest commit", "warn", commit_ok, commit_warn))

    freeze_summary = _load_json(version_root / "validation" / "freeze_audit" / "final_freeze_pytest_summary.json")
    freeze_commit = str(freeze_summary.get("git_commit", ""))
    freeze_status = str(manifest.get("freeze_audit_status", ""))
    freeze_ok = freeze_commit in {current_head, current_short} or freeze_status == "pending_after_cpc_cleanup"
    freeze_warn = [] if freeze_commit in {current_head, current_short} and freeze_status == "current" else [f"freeze_commit={freeze_commit}", f"freeze_audit_status={freeze_status}"]
    checks.append(_check("freeze audit current", "warn", freeze_ok, freeze_warn))

    changelog_text = (root / "CHANGELOG.md").read_text(encoding="utf-8") if (root / "CHANGELOG.md").exists() else ""
    overclaims_closed = "Closed tracked-file leakage" in changelog_text and bool(validation_outputs_tracked)
    checks.append(_check("CHANGELOG consistency", "fail", not overclaims_closed, ["claims closed while validation_outputs is tracked"] if overclaims_closed else []))

    sample_status = str(manifest.get("sample_status", ""))
    sample_status_ok = sample_status in {"template_only_pending_execution", "executed"}
    sample_status_details = [] if sample_status_ok else [f"sample_status={sample_status}"]
    if sample_status == "template_only_pending_execution":
        sample_status_details.append("sample execution pending")
    checks.append(_check("sample execution status", "warn", sample_status_ok, sample_status_details))

    failures = [c for c in checks if c["level"] == "fail" and not c["ok"]]
    warnings = [c for c in checks if c["level"] == "warn" and (not c["ok"] or c["details"])]
    status = "failed" if failures else ("partial" if warnings else "passed")

    payload = {"status": status, "strict": bool(args.strict), "checks": checks}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"CPC readiness: {status}")
        for check in checks:
            if check["level"] == "warn" and check["details"]:
                label = "warn"
            elif check["ok"]:
                label = "ok"
            else:
                label = "fail"
            print(f"- {label}: {check['name']}")
            for item in check["details"]:
                print(f"  - {item}")

    exit_code = 1 if failures or (args.strict and warnings) else 0
    sys.exit(exit_code)
