from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent
RELEASE_ROOT = VERSION_ROOT / "release_package"


def _project_version() -> str:
    with (VERSION_ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_release_version_is_consistent_across_public_metadata() -> None:
    version = _project_version()
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_version = re.search(r'(?m)^version:\s*["\']?([^"\'\s]+)', citation)
    manual = yaml.safe_load((VERSION_ROOT / "docs" / "manual_manifest.yaml").read_text(encoding="utf-8"))
    zenodo = json.loads((REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    codemeta = json.loads((REPO_ROOT / "codemeta.json").read_text(encoding="utf-8"))
    archive = json.loads((RELEASE_ROOT / "archive_manifest.json").read_text(encoding="utf-8"))
    sample = json.loads(
        (RELEASE_ROOT / "sample_output" / "comprehensive_sample_summary.json").read_text(encoding="utf-8")
    )

    assert version == "1.1.0"
    assert citation_version is not None
    assert citation_version.group(1) == version
    assert manual["manual_version"] == manual["package_version"] == version
    assert zenodo["version"] == codemeta["version"] == archive["version"] == version
    assert sample["release_version"] == version
    assert archive["release_tag"] == f"v{version}"


@pytest.mark.packaging
@pytest.mark.release_readiness
def test_sdist_manifest_is_an_explicit_public_whitelist() -> None:
    manifest = (VERSION_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    expected_directives = {
        "include README.md",
        "include LICENSE",
        "include pyproject.toml",
        "include MANIFEST.in",
        "include MANIFEST.md",
        "include USER_MANUAL.md",
        "recursive-include hidden_attractors *.py *.c *.h",
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
    actual_directives = {
        line.strip()
        for line in manifest.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert actual_directives == expected_directives


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_public_release_payload_has_no_case_specific_or_project_plan_narrative() -> None:
    public_files = [
        VERSION_ROOT / "README.md",
        VERSION_ROOT / "MANIFEST.md",
        RELEASE_ROOT / "README_RELEASE.md",
        RELEASE_ROOT / "PROGRAM_SUMMARY.md",
        RELEASE_ROOT / "PUBLISHING_POLICY.md",
        RELEASE_ROOT / "SAMPLE_RUN.md",
        RELEASE_ROOT / "archive_manifest.json",
        RELEASE_ROOT / "sample_input" / "README.md",
        RELEASE_ROOT / "sample_output" / "README.md",
    ]
    blocked_patterns = (
        re.compile(r"\bc[0-9]{3}\b"),
        re.compile(r"\b[a-z]{2}20[0-9]{2}\b"),
        re.compile(r"planned\s+support"),
        re.compile(r"future\s+work"),
    )
    violations = []
    for path in public_files:
        text = path.read_text(encoding="utf-8").lower()
        for pattern in blocked_patterns:
            if pattern.search(text):
                violations.append(
                    f"{path.relative_to(REPO_ROOT).as_posix()}: {pattern.pattern}"
                )
    assert not violations, "\n".join(violations)


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_importable_package_has_no_future_research_placeholders() -> None:
    blocked_fragments = (
        "PLANNED_SEED_FAMILIES",
        "machado_seed_generator",
        "future experiments",
        "Legacy support will be removed in a future version",
        "pyComplexity notebook",
        "intended to grow",
        "future research",
        "future use",
        "coming soon",
        "work in progress",
        "report_targets",
        "by_report",
        "update_report_assets",
    )
    violations = []
    for path in (VERSION_ROOT / "hidden_attractors").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for fragment in blocked_fragments:
            if fragment.lower() in text:
                violations.append(
                    f"{path.relative_to(VERSION_ROOT).as_posix()}: {fragment}"
                )
    assert not violations, "\n".join(violations)


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_project_release_readiness_is_not_in_the_importable_package() -> None:
    source = (
        VERSION_ROOT / "hidden_attractors" / "cli" / "validate.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not any(
        "release" in name.lower() or "submission" in name.lower()
        for name in function_names
    )
    assert "release_package" not in source


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_chaos_validation_records_use_closed_status_vocabulary() -> None:
    validation_root = VERSION_ROOT / "validation" / "chaos_validation"
    blocked_fragments = (
        "pending",
        "future work",
        "not yet validated",
        "must be resolved",
    )
    violations = []
    for suffix in ("*.csv", "*.json", "*.md", "*.yaml", "*.yml"):
        for path in validation_root.rglob(suffix):
            text = path.read_text(encoding="utf-8").lower()
            for fragment in blocked_fragments:
                if fragment in text:
                    violations.append(
                        f"{path.relative_to(VERSION_ROOT).as_posix()}: {fragment}"
                    )
    assert not violations, "\n".join(violations)


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_public_lyapunov_registry_contains_only_callable_methods() -> None:
    from hidden_attractors.analysis.lyapunov_methods import LYAPUNOV_METHODS

    assert LYAPUNOV_METHODS
    assert all(info.implemented for info in LYAPUNOV_METHODS.values())
    assert "fractional_variational_dk2018_block_restart_abm_gs" not in LYAPUNOV_METHODS


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_public_workflow_defaults_do_not_target_validation_assets() -> None:
    from hidden_attractors.workflows.config_loader import _DEFAULTS

    assert _DEFAULTS["system_id"] is None
    assert _DEFAULTS["integrator"] is None
    assert _DEFAULTS["h"] is None
    assert _DEFAULTS["run_seed_search"] is False
    assert _DEFAULTS["run_continuation"] is False
    assert _DEFAULTS["run_final_simulation"] is False
    assert _DEFAULTS["final_simulation"] == {}
    assert _DEFAULTS["hiddenness"] == {}
    assert _DEFAULTS["sphere_tests"] == {"enabled": False}
    assert _DEFAULTS["basin"] == {"enabled": False}
    assert _DEFAULTS["bifurcation"] == {"enabled": False}
    assert _DEFAULTS["validation"]["claims_manifest"] is None
    assert _DEFAULTS["validation"]["fail_on_missing_references"] is False
    assert _DEFAULTS["validation"]["fail_on_unregistered_references"] is False
    assert _DEFAULTS["figures"]["output_root"] == "outputs/figures"
    assert _DEFAULTS["figures"]["export_targets"] == []


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_figure_interception_has_no_implicit_export_target(monkeypatch, tmp_path) -> None:
    import matplotlib.pyplot as plt

    from hidden_attractors.plotting import export as export_module

    captured: dict[str, object] = {}
    staged_pdf = tmp_path / "staged.pdf"
    staged_png = tmp_path / "staged.png"
    staged_pdf.write_bytes(b"pdf")
    staged_png.write_bytes(b"png")

    def fake_export_figure(
        fig,
        figure_id,
        kind,
        metadata_dict,
        run_id="default_run",
        export_targets=None,
    ):
        captured["export_targets"] = export_targets
        return staged_pdf, staged_png

    monkeypatch.setattr(export_module, "export_figure", fake_export_figure)
    fig, _ = plt.subplots()
    try:
        export_module.intercept_and_export_path(
            fig,
            tmp_path / "generic_output.png",
            "characterization",
        )
    finally:
        plt.close(fig)

    assert captured["export_targets"] == []


@pytest.mark.hygiene
@pytest.mark.packaging
@pytest.mark.release_readiness
def test_packaged_config_is_abstract_and_case_free() -> None:
    packaged = VERSION_ROOT / "hidden_attractors" / "configs" / "examples"
    repository = VERSION_ROOT / "configs" / "examples"
    names = {path.name for path in packaged.glob("*.yaml")}
    repository_names = {path.name for path in repository.glob("*.yaml")}

    assert names == {"workflow_contract.yaml"}
    assert repository_names == set()

    path = packaged / "workflow_contract.yaml"
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    assert payload["template"]["artifact_role"] == "public_abstract_workflow_contract"
    assert payload["template"]["runnable"] is False
    assert payload["template"]["scientific_claim"] == "none"
    assert payload["required_inputs"]["system"]["system_id"] is None
    for prohibited in ("chua", "0.9998", "R0_base", "future", "planned"):
        assert prohibited.lower() not in text.lower()


@pytest.mark.hygiene
@pytest.mark.packaging
@pytest.mark.release_readiness
def test_case_specific_workflow_modules_are_not_distributed() -> None:
    assert not (VERSION_ROOT / "hidden_attractors" / "workflows" / "sphere_controls.py").exists()
    assert not (VERSION_ROOT / "hidden_attractors" / "workflows" / "robustness_overlay.py").exists()
    assert not (VERSION_ROOT / "hidden_attractors" / "cli" / "hiddenness.py").exists()
    assert not (VERSION_ROOT / "hidden_attractors" / "cli" / "robustness.py").exists()


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_case_specific_inputs_live_only_under_validation() -> None:
    prohibited_public_inputs = (
        "user_chua_fractional_h001_exploratory.yaml",
        "chua_nonsmooth_exact_danca_non_reproducible.yaml",
        "chua_arctan_wu2023_caputo.json",
        "report_heatmap_inputs.json",
        "unified_caputo_protocol.json",
        "chua_classical_lure_default.yaml",
    )
    configs_root = VERSION_ROOT / "configs"
    leaked = [
        path.relative_to(VERSION_ROOT).as_posix()
        for name in prohibited_public_inputs
        for path in configs_root.rglob(name)
    ]
    leaked.extend(
        path.relative_to(VERSION_ROOT).as_posix()
        for path in configs_root.rglob("*search*.yaml")
    )
    assert not leaked, "\n".join(leaked)


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_superseded_release_reports_are_absent() -> None:
    superseded = (
        "ARCTAN_C590_PROMOTION_BOUNDARY.md",
        "DOCS_COMMAND_SYNC_REPORT.md",
        "FINAL_PYPI_TRACEABILITY_REPORT.md",
        "PYPI_BLOCKING_ITEMS.md",
        "PYPI_RELEASE_CHECKLIST.md",
        "PYPI_RELEASE_REPORT.md",
        "REMAINING_WORK.md",
        "reproducibility_checklist.md",
    )
    present = [name for name in superseded if (RELEASE_ROOT / name).exists()]
    assert not present


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_public_integer_reference_has_no_editorial_export_contract() -> None:
    example_root = VERSION_ROOT / "examples" / "chua_integer_lure_reference"
    config_text = (example_root / "reproducibility.yaml").read_text(
        encoding="utf-8"
    )
    source = (example_root / "run_example.py").read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)

    assert config["validation_role"] == "integer_order_software_control"
    for fragment in (
        "role_in_report",
        "publication_figures",
        "report_assets",
        "library_figures/by_report",
    ):
        assert fragment not in config_text
        assert fragment not in source


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_public_examples_use_generic_library_workflows() -> None:
    examples_root = VERSION_ROOT / "examples"
    blocked_patterns = (
        re.compile(r"\b(?:final|selected)[_-]?candidates?\b"),
        re.compile(r"\brobustness[_ -]?overlays?\b"),
        re.compile(r"\breport[_ -]?assets?\b"),
    )
    public_suffixes = {".ipynb", ".md", ".py", ".yaml", ".yml"}
    violations = []
    for path in examples_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in public_suffixes:
            continue
        searchable = f"{path.name}\n{path.read_text(encoding='utf-8').lower()}"
        for pattern in blocked_patterns:
            if pattern.search(searchable):
                violations.append(
                    f"{path.relative_to(VERSION_ROOT).as_posix()}: "
                    f"{pattern.pattern}"
                )
    assert not violations, "\n".join(violations)


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_notebook_is_generic_characterization_and_not_in_primary_nav() -> None:
    paths = (
        VERSION_ROOT / "docs" / "notebooks.md",
        VERSION_ROOT / "examples" / "notebooks" / "README.md",
        VERSION_ROOT
        / "examples"
        / "notebooks"
        / "hidden_attractors_quickstart.ipynb",
    )
    corpus = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in paths
    )
    for fragment in (
        "load_final_candidate_records",
        "reference candidates",
        "final candidate records",
        "selected_candidates",
    ):
        assert fragment not in corpus

    nav = (VERSION_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Notebooks: notebooks.md" not in nav


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_public_benchmarks_use_synthetic_performance_fixtures() -> None:
    benchmark_root = VERSION_ROOT / "benchmarks"
    source_paths = sorted(benchmark_root.glob("*.py"))
    text_paths = [
        benchmark_root / "README.md",
        VERSION_ROOT / "docs" / "testing_policy.md",
        *source_paths,
    ]
    blocked_patterns = (
        re.compile(r"\b(?:production|campaign|editorial)\b"),
        re.compile(r"\b(?:scientific|validation)\s+reports?\b"),
        re.compile(r"\b(?:compute|runtime)\s+budgets?\b"),
        re.compile(r"\b(?:hundreds|thousands)\b"),
        re.compile(r"\b(?:candidate|parameter)\s+sweeps?\b"),
        re.compile(r"\bextrapolat(?:e|ed|es|ing|ion)\b"),
        re.compile(r"\b(?:future|planned)\s+(?:work|research|support)\b"),
    )
    violations = []
    for path in text_paths:
        text = path.read_text(encoding="utf-8").lower()
        for pattern in blocked_patterns:
            if pattern.search(text):
                violations.append(
                    f"{path.relative_to(VERSION_ROOT).as_posix()}: "
                    f"{pattern.pattern}"
                )

    prohibited_constant_names = {"Q", "H", "LM"}
    prohibited_name_fragments = {
        "budget",
        "campaign",
        "canonical",
        "production",
        "report",
    }
    for path in source_paths:
        source = path.read_text(encoding="utf-8")
        assert "synthetic" in source.lower(), path.relative_to(VERSION_ROOT)
        tree = ast.parse(source)
        assigned_names = set()
        for node in tree.body:
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)

        violations.extend(
            f"{path.relative_to(VERSION_ROOT).as_posix()}: "
            f"prohibited constant name {name}"
            for name in sorted(assigned_names & prohibited_constant_names)
        )
        violations.extend(
            f"{path.relative_to(VERSION_ROOT).as_posix()}: "
            f"prohibited identifier fragment {fragment}"
            for name in sorted(assigned_names)
            for fragment in sorted(prohibited_name_fragments)
            if fragment in name.lower()
        )

    assert not violations, "\n".join(violations)
