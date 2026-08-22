"""Fast contracts for the official MAVPD hidden-chaos example."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "modified_van_der_pol_duffing_integer_hidden_chaos_search"
SCRIPT = EXAMPLE / "run_example.py"
CONFIG = EXAMPLE / "reproducibility.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("mavpd_hidden_chaos_example", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_config_keeps_primary_and_alternative_routes_separate() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["system"]["q"] == 1.0
    assert config["primary_route"]["name"] == "direct_integer_lure"
    assert config["primary_route"]["frequency_grid_used"] is False
    assert config["primary_route"]["fallback_frequency_scan_used"] is False
    assert config["primary_route"]["admissible_omega_min"] == pytest.approx(1.0e-5)
    assert config["primary_route"]["admissible_omega_max"] == pytest.approx(50.0)
    assert config["alternative_parameter_continuation"]["enabled_only_after_primary_route_failure"] is True
    assert config["alternative_parameter_continuation"]["frequency_grid_used"] is False
    assert config["alternative_parameter_continuation"]["xi_step"] == pytest.approx(-0.01)
    assert config["alternative_parameter_continuation"]["screening"]["probes"]["radii"] == pytest.approx(
        [1.0e-7, 1.0e-4]
    )
    alternative = config["alternative_parameter_continuation"]
    assert "selected_offset" not in alternative
    assert alternative["xi_endpoint_role"].startswith("declared local continuation endpoint")
    assert alternative["source_branch_selection_rule"].startswith("lowest direct harmonic frequency")
    assert "zero ambiguous probes" in alternative["selection_rule"]
    assert "zero numerical failures" in alternative["selection_rule"]
    assert config["hiddenness"]["main_contract"].startswith("3 equilibria x 3 radii")
    assert config["source"]["report_values_used_as_search_input"] is False


@pytest.mark.unit
def test_example_exposes_equation_to_seed_contract_without_stored_candidate_input() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "integer_lure_seed(" in source
    assert "nscan=" not in source
    assert 'alternative["selected_offset"]' not in source
    assert "continue_integer_lure_seed(" in source
    assert "continue_integer_parameter_path(" in source
    assert "promote_local_figure_pairs_batch(" in source
    assert 'run_id=run_status["run_id"]' in source
    assert "promote_local_figure_pair(" not in source
    assert "frequency_grid_used_for_seed" in source
    assert 'transfer_convention="c^T (P-s I)^(-1) b' in source
    assert "c^T (s I-A)^(-1) b" not in source
    assert '"selected_candidate_initial_state": selected_states[selected_offset]' in source
    assert 'selection["selected_candidate_initial_state"]' in source
    assert "trajectory initial state differs from selected continuation endpoint" in source
    assert "selected state differs from its parameter-continuation node" in source
    assert 'source_branch_index = int(selection["alternative_source_branch_index"])' in source
    assert "0.1538037983994911" not in source
    assert "0.29604161699400955" not in source


@pytest.mark.integration
def test_contract_only_run_derives_branches_in_temporary_output(tmp_path: Path) -> None:
    module = _module()
    payload = module.run_contract_only(quick=True, output_override=tmp_path)

    assert (tmp_path / "00_system_contract.json").is_file()
    assert len(payload["direct_seed_records"]) == 2
    assert payload["frequency_grid_used_for_seed"] is False
    assert payload["fallback_frequency_scan_used"] is False
    assert payload["report_values_used_as_search_input"] is False
    assert payload["hopf_boundary_at_xi_target"]["values_derived_from_equations"] is True
    assert payload["candidate_parameter_set_published"] is False
    assert len(payload["scientific_source_snapshot"]["files"]) >= 15


@pytest.mark.unit
def test_scientific_source_guard_rejects_a_different_bundle() -> None:
    module = _module()
    bogus = module._scientific_source_snapshot()
    bogus["bundle_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="scientific source bundle changed"):
        module._assert_scientific_sources_unchanged(bogus, phase="unit test")


@pytest.mark.unit
def test_scientific_source_snapshot_detects_new_python_modules(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    for relative in module.SCIENTIFIC_SOURCE_FIXED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixed\n", encoding="utf-8")
    package = tmp_path / "hidden_attractors"
    package.mkdir(exist_ok=True)
    (package / "existing.py").write_text("VALUE = 1\n", encoding="utf-8")
    before = module._scientific_source_snapshot()

    (package / "new_module.py").write_text("VALUE = 2\n", encoding="utf-8")
    after = module._scientific_source_snapshot()

    assert before["bundle_sha256"] != after["bundle_sha256"]
    assert "hidden_attractors/new_module.py" in after["files"]


@pytest.mark.unit
def test_example_figure_export_is_dual_and_title_free(tmp_path: Path) -> None:
    module = _module()
    figure, axis = module.plt.subplots()
    axis.plot([0.0, 1.0], [0.0, 1.0])
    axis.set_title("must be removed")
    figure.suptitle("must also be removed")
    target = tmp_path / "figure.png"

    module._save_figure_pair(figure, target)
    module.plt.close(figure)

    assert target.is_file()
    assert target.with_suffix(".pdf").is_file()


@pytest.mark.unit
def test_completed_run_promotes_all_figures_with_its_exact_run_id(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    import hidden_attractors.plotting.export as export_mod
    import hidden_attractors.plotting.manifest as manifest_mod

    library_root = tmp_path / "library_figures"
    monkeypatch.setattr(export_mod, "LIBRARY_FIGURES_ROOT", library_root)
    monkeypatch.setattr(manifest_mod, "LIBRARY_FIGURES_ROOT", library_root)
    output = tmp_path / "run"
    output.mkdir()
    config = module.load_config(quick=False)
    snapshot = module._scientific_source_snapshot()
    status = module._new_run_status(output, config, snapshot)
    figures = output / "figures"
    figures.mkdir()
    for figure_id in module.FIGURE_DATA_SOURCES:
        (figures / f"{figure_id}.png").write_bytes(f"png:{figure_id}".encode())
        (figures / f"{figure_id}.pdf").write_bytes(f"pdf:{figure_id}".encode())
    rows = module._finalize_figure_manifest(
        output,
        config,
        snapshot,
        {"gamma": 0.15, "delta": 100.0, "rho": 200.0, "xi": 2.85},
        run_id=status["run_id"],
        promote=True,
    )
    timings = [
        {"phase": "candidate_gate", "seconds": 1.0, "timing_source": "perf_counter"},
        {"phase": "total", "seconds": 2.0, "timing_source": "perf_counter"},
    ]
    module._write_rows(output / "phase_timings.csv", timings)
    manifest = {"run_id": status["run_id"], "figures": rows, "timings": timings}
    module._write_json(output / "run_manifest.json", manifest)
    module._record_run_phase(
        output,
        status,
        "manifest",
        ("figures/figure_manifest.json", "phase_timings.csv", "run_manifest.json"),
    )
    status["status"] = "complete"
    module._write_json(output / "run_status.json", status)

    promoted_rows, updated = module._promote_completed_figures(
        output, rows, manifest, status, snapshot
    )

    assert all(row["promoted_to_global_manifest"] for row in promoted_rows)
    assert {row["run_id"] for row in promoted_rows} == {status["run_id"]}
    assert all(
        path.startswith("library_figures/")
        and not Path(path).is_absolute()
        and "\\" not in path
        for row in promoted_rows
        for path in row["central_paths"].values()
    )
    assert updated["global_figure_promotion"]["global_manifest_paths"] == [
        "library_figures/manifests/figure_manifest.json",
        "library_figures/manifests/figure_manifest.csv",
    ]
    assert updated["global_figure_promotion"]["run_id"] == status["run_id"]
    global_manifest = json.loads(
        (library_root / "manifests" / "figure_manifest.json").read_text(encoding="utf-8")
    )
    assert {entry["run_id"] for entry in global_manifest} == {status["run_id"]}


@pytest.mark.unit
def test_quick_default_output_is_isolated_and_resume_is_rejected() -> None:
    module = _module()
    quick = module.load_config(quick=True)
    full = module.load_config(quick=False)

    assert module._output_dir(quick, None) != module._output_dir(full, None)
    assert "tmp" in module._output_dir(quick, None).parts
    assert "tmp" in module._output_dir(full, None).parts
    with pytest.raises(SystemExit):
        module.main(["--quick", "--resume-validated-candidate"])


@pytest.mark.unit
def test_contract_only_default_output_is_always_isolated_from_canonical() -> None:
    module = _module()
    canonical = module._canonical_output_dir(module.load_config(quick=False))

    quick_output = module.ROOT / "tmp" / "mavpd_integer_hidden_chaos_contract_quick"
    full_output = module.ROOT / "tmp" / "mavpd_integer_hidden_chaos_contract_full"

    assert quick_output != canonical
    assert full_output != canonical

    with pytest.raises(ValueError, match="isolated from the canonical evidence"):
        module.run_contract_only(quick=False, output_override=canonical)


@pytest.mark.unit
def test_run_status_binds_artifacts_and_detects_mutation(tmp_path: Path) -> None:
    module = _module()
    config = module.load_config(quick=False)
    snapshot = module._scientific_source_snapshot()
    status = module._new_run_status(tmp_path, config, snapshot)
    second_status = module._new_run_status(tmp_path, config, snapshot)
    assert status["run_id"] != second_status["run_id"]
    status = second_status
    assert set(status["runtime_environment"]) == {
        "python_version",
        "python_implementation",
        "platform",
        "numpy_version",
        "scipy_version",
    }
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")

    module._record_run_phase(tmp_path, status, "unit", ("artifact.json",))
    module._verify_recorded_artifacts(tmp_path, status, ("artifact.json",))
    artifact.write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        module._verify_recorded_artifacts(tmp_path, status, ("artifact.json",))


@pytest.mark.unit
def test_full_run_rejects_a_nonempty_staging_directory(tmp_path: Path) -> None:
    module = _module()
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "stale.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="run output is not empty"):
        module.run_all(quick=True, output_override=occupied)


@pytest.mark.unit
def test_probe_coverage_requires_each_equilibrium_radius_direction_cell() -> None:
    module = _module()
    probes = [
        SimpleNamespace(equilibrium=equilibrium, radius=radius, direction_id=direction, status="ok")
        for equilibrium in ("E0", "E+")
        for radius in (1.0e-3, 1.0e-5)
        for direction in ("d0", "d1")
    ]

    complete = module._probe_cell_coverage(
        probes,
        equilibrium_names=("E0", "E+"),
        radii=(1.0e-3, 1.0e-5),
        expected_per_cell=2,
    )
    missing = module._probe_cell_coverage(
        probes[:-1],
        equilibrium_names=("E0", "E+"),
        radii=(1.0e-3, 1.0e-5),
        expected_per_cell=2,
    )
    duplicate_direction = [*probes[:-1], SimpleNamespace(equilibrium="E+", radius=1.0e-5, direction_id="d0", status="ok")]
    duplicated = module._probe_cell_coverage(
        duplicate_direction,
        equilibrium_names=("E0", "E+"),
        radii=(1.0e-3, 1.0e-5),
        expected_per_cell=2,
    )

    assert complete["complete"] is True
    assert missing["complete"] is False
    assert duplicated["complete"] is False


@pytest.mark.unit
def test_resume_uses_persisted_timers_without_file_mtime_reconstruction(tmp_path: Path) -> None:
    module = _module()
    timing = tmp_path / "phase_timings.csv"
    timing.write_text(
        "phase,seconds\ncontract,1.25\nsearch,2.5\ndiagnostics,3.75\nhiddenness,9\ntotal,20\n",
        encoding="utf-8",
    )

    rows = module._load_persisted_pre_resume_timings(timing)

    assert [row["phase"] for row in rows] == ["contract", "search", "diagnostics"]
    assert [row["seconds"] for row in rows] == pytest.approx([1.25, 2.5, 3.75])
    assert all(row["timing_source"] == "persisted_perf_counter_from_interrupted_run" for row in rows)
