from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "local_reports"
    / "comparacion_dynamicalsystems_hafo_enteros"
    / "refresh_mavpd_chaos_evidence.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("refresh_mavpd_chaos_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def refresh_module() -> ModuleType:
    return _load_script()


@pytest.fixture(autouse=True)
def _stub_authoritative_validation(
    monkeypatch: pytest.MonkeyPatch,
    refresh_module: ModuleType,
) -> None:
    def validate(
        run_dir: Path,
        repo_root: Path,
        scientific_source_root: Path,
        *,
        require_active_promotion: bool,
    ) -> dict[str, object]:
        del repo_root, scientific_source_root
        assert require_active_promotion is True
        status = json.loads((Path(run_dir) / "run_status.json").read_text(encoding="utf-8"))
        return {
            "validator": "mavpd_integer_hidden_chaos_run",
            "ok": True,
            "active_promotion_required": True,
            "active_promotion_checked": True,
            "run_id": status["run_id"],
            "config_sha256": status["config_sha256"],
            "scientific_source_bundle_sha256": status["scientific_source_snapshot"]["bundle_sha256"],
        }

    monkeypatch.setattr(
        refresh_module,
        "_load_authoritative_validator",
        lambda _repo_root: validate,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scientific_snapshot() -> dict[str, object]:
    files = {
        "examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/run_example.py": "d" * 64,
        "examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/reproducibility.yaml": "e" * 64,
        "validation/wolfram/cases/mavpd_integer.wl": "f" * 64,
    }
    material = "".join(
        f"{path}\0{digest}\n" for path, digest in sorted(files.items())
    ).encode()
    return {
        "algorithm": "sha256",
        "bundle_sha256": hashlib.sha256(material).hexdigest(),
        "files": files,
    }


def _rebind_artifact(run_dir: Path, relative: str) -> None:
    status_path = run_dir / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["artifacts"][relative] = _sha256(run_dir / relative)
    _write_json(status_path, status)


def _make_full_run(path: Path, module: ModuleType) -> dict[str, object]:
    snapshot = _scientific_snapshot()
    bundle = str(snapshot["bundle_sha256"])
    run_id = (
        "mavpd_integer_hidden_chaos-full-20260803T120000000000Z-"
        f"{bundle[:12]}-{'c' * 12}"
    )
    config_sha = "b" * 64
    runtime = {
        "python_version": "3.14.3",
        "python_implementation": "CPython",
        "platform": "Windows-test",
        "numpy_version": "2.4.0",
        "scipy_version": "1.17.0",
    }
    path.mkdir(parents=True)

    figure_rows = []
    candidate_parameters = {
        "xi": 2.85,
        "gamma": 0.15380379839949113,
        "delta": 100,
        "rho": 200,
    }
    for figure_id in module.ALL_FIGURE_IDS:
        hashes: dict[str, str] = {}
        for extension in ("png", "pdf"):
            figure_path = path / "figures" / f"{figure_id}.{extension}"
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            figure_path.write_bytes(f"{figure_id}-{extension}-fixture\n".encode())
            hashes[extension] = _sha256(figure_path)
        figure_rows.append(
            {
                "figure_id": figure_id,
                "local_png": f"figures/{figure_id}.png",
                "local_pdf": f"figures/{figure_id}.pdf",
                "png_sha256": hashes["png"],
                "pdf_sha256": hashes["pdf"],
                "run_id": run_id,
                "global_promotion_requested": True,
                "promoted_to_global_manifest": True,
                "central_paths": {"png": "global/example.png", "pdf": "global/example.pdf"},
                "metadata": {
                    "quick_smoke_only": False,
                    "scientific_source_bundle_sha256": bundle,
                    "parameters": (
                        {"xi": 3.1, "gamma": 0.1, "delta": 100, "rho": 200}
                        if figure_id.startswith("00_")
                        else candidate_parameters
                    ),
                },
            }
        )

    return_map = [
        {
            "coordinate": 0,
            "K": 0.9985162431217313,
            "state": "zero_one_chaotic_candidate",
            "zero_one_alone_does_not_certify_chaos": True,
            "chaos_certified_by_zero_one": False,
            "hiddenness_certified_by_zero_one": False,
        },
        {
            "coordinate": 2,
            "K": 0.9984323397081207,
            "state": "zero_one_chaotic_candidate",
            "zero_one_alone_does_not_certify_chaos": True,
            "chaos_certified_by_zero_one": False,
            "hiddenness_certified_by_zero_one": False,
        },
    ]
    strict = [0.7133503984936994, -0.0009058832510818009, -20.923624956275255]
    control = [0.6830338121422247, 0.0011359817372712411, -20.898002850234473]
    divergence_residual = -0.0000421357
    mean_divergence = sum(strict) - divergence_residual
    diagnostics = {
        "candidate_parameters": candidate_parameters,
        "lyapunov": {"status": "ok", "exponents": strict},
        "lyapunov_control": {"status": "ok", "exponents": control},
        "mean_vector_field_divergence": mean_divergence,
        "lyapunov_sum_minus_mean_divergence": divergence_residual,
        "kaplan_yorke_dimension": 2.034049765121074,
        "poincare": {"crossing_count": 1174},
        "zero_one": {"return_map_results": return_map},
        "efork_crosscheck": {
            "status": "ok",
            "target_match": {
                "classification": "same_attractor_under_calibrated_cloud_test"
            },
        },
        "finite_time_only": True,
    }
    main_by_equilibrium = {
        label: {
            "n": 36,
            "target_hits": 0,
            "ambiguous": 0,
            "numerical_failures": 0,
            "equilibrium_destinations": 36,
        }
        for label in ("E0", "E+", "E-")
    }
    hiddenness = {
        "main": {
            "n_probes": 108,
            "target_hits": 0,
            "ambiguous": 0,
            "numerical_failures": 0,
            "tested_all_declared_equilibria": True,
            "by_equilibrium": main_by_equilibrium,
            "sampled_hiddenness_status": "hidden_under_tested_neighborhoods",
            "finite_sample_only": True,
            "global_hiddenness_proved": False,
        },
        "targeted_E0_unstable_direction": {
            "n_probes": 4,
            "target_hits": 0,
            "ambiguous": 0,
            "numerical_failures": 0,
            "by_equilibrium": {
                "E0": {
                    "n": 4,
                    "target_hits": 0,
                    "ambiguous": 0,
                    "numerical_failures": 0,
                    "equilibrium_destinations": 4,
                }
            },
            "sampled_hiddenness_status": "hidden_under_tested_neighborhoods",
            "finite_sample_only": True,
            "global_hiddenness_proved": False,
        },
        "n_probes": 112,
        "target_hits": 0,
        "ambiguous": 0,
        "numerical_failures": 0,
        "coverage_by_equilibrium_radius": {
            "main": {"complete": True},
            "targeted_E0_unstable_direction": {"complete": True},
            "complete": True,
        },
        "sampled_hiddenness_status": "hidden_under_tested_neighborhoods",
        "finite_sample_only": True,
        "global_hiddenness_proved": False,
    }
    gate = {
        "attractor_status": "hidden_under_tested_neighborhoods",
        "chaotic_hidden_promotion_allowed": True,
        "hiddenness_promotion_allowed": True,
        "promotion_allowed": True,
        "hidden_chaos_status": "chaotic_hidden_under_tested_neighborhoods",
        "quick_smoke_only": False,
        "warnings": [],
        "missing_conditions": [],
    }
    robustness = {
        "tested_h": True,
        "tested_memory": False,
        "memory_applicable": False,
        "tested_t_final": True,
        "tested_integrator": True,
        "integrator_match": True,
        "consistent": True,
    }
    receipt = {
        "status": "committed",
        "run_id": run_id,
        "scientific_source_bundle_sha256": bundle,
        "figure_count": len(figure_rows),
        "figure_ids": list(module.ALL_FIGURE_IDS),
        "seconds": 0.25,
        "global_manifest_paths": ["outputs/library_figures/manifests/figure_manifest.json"],
    }
    hopf = 0.14380379839949112
    screening_specs = (
        (0.002, 0.92911, 6, False),
        (0.003, 0.89340, 6, False),
        (0.005, 0.18832, 6, False),
        (0.008, 0.81221, 0, True),
        (0.010, 0.72394, 0, True),
        (0.012, 0.31512, 0, False),
        (0.015, -0.00887, 0, False),
    )
    screening = []
    screening_probes = []
    for offset, lambda_one, hit_count, eligible in screening_specs:
        gamma = hopf + offset
        screening.append(
            {
                "hopf_offset": offset,
                "gamma": gamma,
                "equilibrium_stability_margin": 0.05,
                "lambda_1": lambda_one,
                "lambda_2": 0.01,
                "lambda_3": -20.8,
                "lyapunov_status": "ok",
                "E0_probe_count": 12,
                "E0_target_hits": hit_count,
                "E0_ambiguous": 0,
                "eligible_hidden_chaos_screen": eligible,
            }
        )
        for sample_id in range(12):
            target_hit = sample_id < hit_count
            screening_probes.append(
                {
                    "hopf_offset": offset,
                    "gamma": gamma,
                    "contract": "candidate_screen_E0",
                    "sample_id": sample_id,
                    "equilibrium": "E0",
                    "status": "ok",
                    "destination": (
                        "same_attractor_under_calibrated_cloud_test"
                        if target_hit
                        else "equilibrium_E0"
                    ),
                    "target_hit": target_hit,
                    "ambiguous": False,
                }
            )
    selected_screen = dict(screening[4])
    selection = {
        "hopf_boundary": hopf,
        "selected": selected_screen,
        "scientific_source_snapshot": snapshot,
        "no_frequency_sweep": True,
        "primary_route_chaos_screen_failed_before_alternative": True,
    }
    timings = [
        {"phase": "contract", "seconds": 0.6430566000053659, "timing_source": "perf_counter"},
        {"phase": "search", "seconds": 52.39817610004684, "timing_source": "perf_counter"},
        {"phase": "diagnostics", "seconds": 35.780649700027425, "timing_source": "perf_counter"},
        {"phase": "hiddenness", "seconds": 33.7186953999917, "timing_source": "perf_counter"},
        {"phase": "candidate_gate", "seconds": 0.7487727000261657, "timing_source": "perf_counter"},
        {"phase": "total", "seconds": 123.53241159999743, "timing_source": "perf_counter"},
    ]
    manifest = {
        "case_id": module.CASE_ID,
        "run_id": run_id,
        "config_sha256": config_sha,
        "runtime_environment": runtime,
        "quick_mode": False,
        "candidate_parameters": diagnostics["candidate_parameters"],
        "hopf_boundary": hopf,
        "selected_hopf_offset": 0.01,
        "lyapunov_exponents": diagnostics["lyapunov"]["exponents"],
        "scientific_source_snapshot": snapshot,
        "hiddenness": hiddenness,
        "candidate_gate": gate,
        "figures": figure_rows,
        "global_figure_promotion": receipt,
        "frequency_grid_used_for_search": False,
        "timings": timings,
    }
    gate_payload = {
        "gate": gate,
        "evidence": {
            "run_metadata": {
                "run_id": run_id,
                "parameters": candidate_parameters,
                "provenance": {"scientific_source_snapshot": snapshot},
            },
            "lyapunov": {"exponents": strict},
            "zero_one": {"K": sum(row["K"] for row in return_map) / 2},
            "hiddenness": {
                "target_hits_from_equilibria": 0,
                "numerical_failures": 0,
                "basin_controls_complete": True,
                "coverage_by_equilibrium_radius_complete": True,
            },
            "robustness": robustness,
        },
    }
    stride_states = (
        ["zero_one_chaotic_candidate"] * 3
        + ["zero_one_regular_candidate"]
        + ["zero_one_inconclusive"] * 6
    )
    stride_rows = [
        {
            "series": "flow_y1",
            "stride": stride,
            "effective_sample_step": 0.02 * stride,
            "samples": 1000,
            "K": 0.8,
            "state": state,
        }
        for stride, state in zip((5, 10, 15, 20, 25, 30, 40, 50, 75, 100), stride_states)
    ]
    hiddenness_probes = []
    for sample_id in range(108):
        equilibrium = ("E0", "E+", "E-")[sample_id // 36]
        hiddenness_probes.append(
            {
                "contract": "main_3x3xN",
                "sample_id": sample_id,
                "equilibrium": equilibrium,
                "status": "ok",
                "destination": f"equilibrium_{equilibrium}",
                "target_hit": False,
                "ambiguous": False,
                "closest_equilibrium": equilibrium,
            }
        )
    hiddenness_probes.extend(
        {
            "contract": "targeted_E0_unstable",
            "sample_id": sample_id,
            "equilibrium": "E0",
            "status": "ok",
            "destination": "equilibrium_E0",
            "target_hit": False,
            "ambiguous": False,
            "closest_equilibrium": "E0",
        }
        for sample_id in range(4)
    )

    _write_json(
        path / "00_system_contract.json",
        {
            "scientific_source_snapshot": snapshot,
            "base_parameters": {"xi": 3.1, "gamma": 0.1, "delta": 100, "rho": 200},
            "hopf_boundary_at_xi_target": {"selected_high_gamma_boundary": hopf},
            "frequency_grid_used_for_seed": False,
            "fallback_frequency_scan_used": False,
            "report_values_used_as_search_input": False,
        },
    )
    _write_csv(path / "03_candidate_screening.csv", screening)
    _write_csv(path / "03_candidate_screening_probes.csv", screening_probes)
    _write_json(
        path / "03_candidate_screening_contract.json",
        {"finite_sample_only": True},
    )
    _write_json(path / "03_candidate_selection.json", selection)
    _write_json(path / "05_chaos_diagnostics.json", diagnostics)
    _write_csv(path / "05_zero_one_stride_sensitivity.csv", stride_rows)
    _write_json(path / "05_zero_one_return_map.json", return_map)
    _write_csv(path / "07_hiddenness_probes.csv", hiddenness_probes)
    _write_json(path / "07_hiddenness_summary.json", hiddenness)
    _write_json(path / "08_robustness_matrix.json", robustness)
    _write_json(path / "09_candidate_gate.json", gate_payload)
    _write_csv(path / "phase_timings.csv", timings)
    _write_json(path / "figures" / "figure_manifest.json", {"figures": figure_rows})
    _write_json(path / "figures" / "global_promotion_receipt.json", receipt)
    _write_json(path / "run_manifest.json", manifest)

    artifacts = {relative: _sha256(path / relative) for relative in module.REQUIRED_LEDGER_PATHS}
    status = {
        "status": "complete",
        "quick_mode": False,
        "run_id": run_id,
        "config_sha256": config_sha,
        "runtime_environment": runtime,
        "scientific_source_snapshot": snapshot,
        "completed_phases": list(module.DIRECT_COMPLETED_PHASES),
        "last_completed_phase": "global_figure_promotion",
        "completed_at_utc": "2026-08-03T18:00:00+00:00",
        "artifacts": artifacts,
    }
    _write_json(path / "run_status.json", status)
    return {
        "run_id": run_id,
        "bundle": bundle,
        "status": status,
        "manifest": manifest,
        "gate_payload": gate_payload,
        "contract": {"scientific_source_snapshot": snapshot},
    }


def test_refreshes_exact_macros_rows_and_transactional_asset_pairs(
    tmp_path: Path, refresh_module: ModuleType
) -> None:
    run_dir = tmp_path / "run"
    report_dir = tmp_path / "report"
    fixture = _make_full_run(run_dir, refresh_module)

    result = refresh_module.refresh_evidence(run_dir, report_dir)

    assert result == {
        "status": "refreshed",
        "run_id": fixture["run_id"],
        "scientific_source_bundle_sha256": fixture["bundle"],
        "report_dir": str(report_dir.resolve()),
        "asset_pairs": 7,
        "mavpd_macro_count": 22,
        "screening_rows": 7,
        "report_date": refresh_module.DEFAULT_REPORT_DATE,
        "authoritative_validation": "passed_with_active_promotion",
    }
    assert not (report_dir / ".mavpd_chaos_refresh.lock").exists()
    assert not list(report_dir.glob(".mavpd-refresh-*"))
    for figure_id, destination_stem in refresh_module.FIGURE_ASSETS:
        for extension in ("png", "pdf"):
            assert (
                report_dir / "assets" / f"{destination_stem}.{extension}"
            ).read_bytes() == (run_dir / "figures" / f"{figure_id}.{extension}").read_bytes()

    generated = (report_dir / "mavpd_chaos_generated.tex").read_text(encoding="utf-8")
    assert generated.count(r"\newcommand{\MAVPDChaos") == 22
    assert r"\renewcommand{\FechaCorteDatos}{3 de agosto de 2026}" in generated
    assert r"\newcommand{\MAVPDChaosLEOne}{0.7133503984936994}" in generated
    assert r"\newcommand{\MAVPDChaosMainProbes}{108}" in generated
    assert r"\newcommand{\MAVPDChaosTotalTime}{123.53241159999743}" in generated

    rows = (report_dir / "mavpd_chaos_screening_rows.tex").read_text(encoding="utf-8")
    assert sum(line.endswith(r"\\") for line in rows.splitlines()) == 7
    assert rows.count("Elegible y seleccionada por la regla declarada.") == 1
    assert "0.010 & 0.72394 & 0/12" in rows
    assert "transición" not in rows.lower()

    provenance = json.loads(
        (report_dir / "assets" / "mavpd_chaos_assets_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["run_id"] == fixture["run_id"]
    assert provenance["scientific_source_bundle_sha256"] == fixture["bundle"]
    assert provenance["config_sha256"] == "b" * 64
    assert provenance["completed_at_utc"] == "2026-08-03T18:00:00+00:00"
    assert provenance["refresh_transaction"]["filesystem_batch_atomic"] is False
    assert len(provenance["assets"]) == 7
    for asset in provenance["assets"]:
        for extension in ("png", "pdf"):
            record = asset["formats"][extension]
            assert record["source"]["sha256"] == record["destination"]["sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("quick", "quick/smoke"),
        ("incomplete", "must be 'complete'"),
        ("gate_failed", "chaotic_hidden_promotion_allowed"),
        ("run_id", "inconsistent run_id"),
        ("source_bundle", "bundle_sha256 does not match"),
    ],
)
def test_refuses_nonpromotable_or_inconsistent_runs_before_writing(
    tmp_path: Path,
    refresh_module: ModuleType,
    mutation: str,
    message: str,
) -> None:
    run_dir = tmp_path / "run"
    report_dir = tmp_path / "report"
    _make_full_run(run_dir, refresh_module)
    sentinel = report_dir / "assets" / "mavpd_chaos_screen.png"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"existing-report-asset")

    if mutation in {"quick", "incomplete"}:
        status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
        if mutation == "quick":
            status["quick_mode"] = True
        else:
            status["status"] = "in_progress"
        _write_json(run_dir / "run_status.json", status)
    elif mutation == "gate_failed":
        payload = json.loads((run_dir / "09_candidate_gate.json").read_text(encoding="utf-8"))
        payload["gate"]["chaotic_hidden_promotion_allowed"] = False
        _write_json(run_dir / "09_candidate_gate.json", payload)
        _rebind_artifact(run_dir, "09_candidate_gate.json")
    elif mutation == "run_id":
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        manifest["run_id"] = "different-run"
        _write_json(run_dir / "run_manifest.json", manifest)
        _rebind_artifact(run_dir, "run_manifest.json")
    elif mutation == "source_bundle":
        contract = json.loads((run_dir / "00_system_contract.json").read_text(encoding="utf-8"))
        contract["scientific_source_snapshot"]["bundle_sha256"] = "c" * 64
        _write_json(run_dir / "00_system_contract.json", contract)
        _rebind_artifact(run_dir, "00_system_contract.json")
    else:  # pragma: no cover - guards the parametrization itself
        raise AssertionError(mutation)

    with pytest.raises(refresh_module.EvidenceRefreshError, match=message):
        refresh_module.refresh_evidence(run_dir, report_dir)
    assert sentinel.read_bytes() == b"existing-report-asset"
    assert not (report_dir / "mavpd_chaos_generated.tex").exists()


def test_refuses_ledger_hash_mismatch(tmp_path: Path, refresh_module: ModuleType) -> None:
    run_dir = tmp_path / "run"
    _make_full_run(run_dir, refresh_module)
    diagnostics = json.loads((run_dir / "05_chaos_diagnostics.json").read_text(encoding="utf-8"))
    diagnostics["kaplan_yorke_dimension"] = 2.5
    _write_json(run_dir / "05_chaos_diagnostics.json", diagnostics)

    with pytest.raises(refresh_module.EvidenceRefreshError, match="artifact hash mismatch"):
        refresh_module.refresh_evidence(run_dir, tmp_path / "report")


def test_cli_reports_success_for_explicit_report_dir(
    tmp_path: Path, refresh_module: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    report_dir = tmp_path / "report"
    _make_full_run(run_dir, refresh_module)

    assert refresh_module.main(["--run-dir", str(run_dir), "--report-dir", str(report_dir)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "refreshed"
    assert output["asset_pairs"] == 7


def test_refuses_failed_authoritative_validation_before_writing(
    tmp_path: Path,
    refresh_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    report_dir = tmp_path / "report"
    _make_full_run(run_dir, refresh_module)
    sentinel = report_dir / "assets" / "mavpd_chaos_screen.png"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"unchanged")
    monkeypatch.setattr(
        refresh_module,
        "_run_authoritative_validation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            refresh_module.EvidenceRefreshError("authoritative MAVPD validation failed")
        ),
    )

    with pytest.raises(refresh_module.EvidenceRefreshError, match="authoritative"):
        refresh_module.refresh_evidence(run_dir, report_dir)
    assert sentinel.read_bytes() == b"unchanged"


def test_requires_authoritative_active_promotion_check(
    tmp_path: Path,
    refresh_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _make_full_run(run_dir, refresh_module)

    def validator(*_args: object, **_kwargs: object) -> dict[str, object]:
        status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
        return {
            "validator": "mavpd_integer_hidden_chaos_run",
            "ok": True,
            "active_promotion_required": True,
            "active_promotion_checked": False,
            "run_id": status["run_id"],
            "config_sha256": status["config_sha256"],
            "scientific_source_bundle_sha256": status["scientific_source_snapshot"]["bundle_sha256"],
        }

    monkeypatch.setattr(
        refresh_module,
        "_load_authoritative_validator",
        lambda _repo_root: validator,
    )
    with pytest.raises(refresh_module.EvidenceRefreshError, match="active global promotion"):
        refresh_module.refresh_evidence(run_dir, tmp_path / "report")


def test_refuses_resumed_run_as_timing_evidence(
    tmp_path: Path, refresh_module: ModuleType
) -> None:
    run_dir = tmp_path / "run"
    _make_full_run(run_dir, refresh_module)
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    status["completed_phases"][3:6] = [
        "hiddenness_resumed",
        "candidate_gate_and_figures_resumed",
        "manifest_resumed",
    ]
    _write_json(run_dir / "run_status.json", status)

    with pytest.raises(refresh_module.EvidenceRefreshError, match="uninterrupted full-run"):
        refresh_module.refresh_evidence(run_dir, tmp_path / "report")


def test_refuses_seven_figure_receipt_even_though_report_copies_seven(
    tmp_path: Path, refresh_module: ModuleType
) -> None:
    run_dir = tmp_path / "run"
    _make_full_run(run_dir, refresh_module)
    receipt_path = run_dir / "figures" / "global_promotion_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["figure_count"] = 7
    receipt["figure_ids"] = receipt["figure_ids"][1:]
    _write_json(receipt_path, receipt)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["global_figure_promotion"] = receipt
    _write_json(manifest_path, manifest)
    _rebind_artifact(run_dir, "figures/global_promotion_receipt.json")
    _rebind_artifact(run_dir, "run_manifest.json")

    with pytest.raises(refresh_module.EvidenceRefreshError, match="all eight figures"):
        refresh_module.refresh_evidence(run_dir, tmp_path / "report")


def test_refuses_non_equilibrium_destination_behind_zero_hit_summary(
    tmp_path: Path, refresh_module: ModuleType
) -> None:
    run_dir = tmp_path / "run"
    _make_full_run(run_dir, refresh_module)
    probe_path = run_dir / "07_hiddenness_probes.csv"
    with probe_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["destination"] = "different_from_target_under_calibrated_cloud_test"
    _write_csv(probe_path, rows)
    _rebind_artifact(run_dir, "07_hiddenness_probes.csv")

    with pytest.raises(refresh_module.EvidenceRefreshError, match="equilibrium destinations"):
        refresh_module.refresh_evidence(run_dir, tmp_path / "report")


def test_refuses_inconsistent_gamma_offset_and_kaplan_yorke(
    tmp_path: Path, refresh_module: ModuleType
) -> None:
    run_dir = tmp_path / "run"
    _make_full_run(run_dir, refresh_module)
    selection_path = run_dir / "03_candidate_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["selected"]["gamma"] += 0.001
    _write_json(selection_path, selection)
    _rebind_artifact(run_dir, "03_candidate_selection.json")
    with pytest.raises(refresh_module.EvidenceRefreshError, match="selected gamma"):
        refresh_module.refresh_evidence(run_dir, tmp_path / "report-gamma")

    # Restore a fresh fixture so the second rejection exercises the derived metric.
    second_run = tmp_path / "run-ky"
    _make_full_run(second_run, refresh_module)
    diagnostics_path = second_run / "05_chaos_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["kaplan_yorke_dimension"] = 2.5
    _write_json(diagnostics_path, diagnostics)
    _rebind_artifact(second_run, "05_chaos_diagnostics.json")
    with pytest.raises(refresh_module.EvidenceRefreshError, match="Kaplan-Yorke"):
        refresh_module.refresh_evidence(second_run, tmp_path / "report-ky")


def test_cached_figure_bytes_are_used_after_source_mutation(
    tmp_path: Path,
    refresh_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    report_dir = tmp_path / "report"
    _make_full_run(run_dir, refresh_module)
    source = run_dir / "figures" / "03_continuation_screen.png"
    original = source.read_bytes()
    original_commit = refresh_module._commit_transaction

    def mutate_then_commit(report: Path, outputs: dict[Path, bytes]) -> None:
        source.write_bytes(b"mutated-after-validation")
        original_commit(report, outputs)

    monkeypatch.setattr(refresh_module, "_commit_transaction", mutate_then_commit)
    refresh_module.refresh_evidence(run_dir, report_dir)
    assert (report_dir / "assets" / "mavpd_chaos_screen.png").read_bytes() == original


def test_postverification_failure_rolls_back_every_output(
    tmp_path: Path,
    refresh_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    report_dir = tmp_path / "report"
    _make_full_run(run_dir, refresh_module)
    existing_asset = report_dir / "assets" / "mavpd_chaos_screen.png"
    existing_tex = report_dir / "mavpd_chaos_generated.tex"
    existing_asset.parent.mkdir(parents=True)
    existing_asset.write_bytes(b"old-asset")
    existing_tex.write_bytes(b"old-tex")

    def fail_verification(_outputs: dict[Path, bytes]) -> None:
        raise refresh_module.EvidenceRefreshError("injected post-write verification failure")

    monkeypatch.setattr(refresh_module, "_verify_output_payloads", fail_verification)
    with pytest.raises(refresh_module.EvidenceRefreshError, match="injected"):
        refresh_module.refresh_evidence(run_dir, report_dir)
    assert existing_asset.read_bytes() == b"old-asset"
    assert existing_tex.read_bytes() == b"old-tex"
    assert not (report_dir / "mavpd_chaos_screening_rows.tex").exists()


def test_existing_refresh_lock_is_fail_closed(
    tmp_path: Path, refresh_module: ModuleType
) -> None:
    run_dir = tmp_path / "run"
    report_dir = tmp_path / "report"
    _make_full_run(run_dir, refresh_module)
    report_dir.mkdir()
    (report_dir / ".mavpd_chaos_refresh.lock").write_text("held\n", encoding="utf-8")

    with pytest.raises(refresh_module.EvidenceRefreshError, match="holds the lock"):
        refresh_module.refresh_evidence(run_dir, report_dir)


def test_provenance_uses_repo_relative_path_and_explicit_date(
    tmp_path: Path, refresh_module: ModuleType
) -> None:
    run_dir = tmp_path / "validation" / "reference_cases" / refresh_module.CASE_ID
    report_dir = tmp_path / "report"
    _make_full_run(run_dir, refresh_module)
    refresh_module.refresh_evidence(
        run_dir,
        report_dir,
        repo_root=tmp_path,
        scientific_source_root=tmp_path,
        report_date="4 de agosto de 2026",
    )
    provenance = json.loads(
        (report_dir / "assets" / "mavpd_chaos_assets_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["source_run_directory"] == (
        "validation/reference_cases/mavpd_integer_hidden_chaos"
    )
    assert provenance["source_run_directory_kind"] == "repo_relative"
    assert provenance["report_date"] == "4 de agosto de 2026"
    generated = (report_dir / "mavpd_chaos_generated.tex").read_text(encoding="utf-8")
    assert r"\renewcommand{\FechaCorteDatos}{4 de agosto de 2026}" in generated


def test_tex_numbers_never_emit_raw_exponent_and_date_is_single_line(
    refresh_module: ModuleType,
) -> None:
    assert refresh_module._number_text("1e-7", "small number") == "0.0000001"
    with pytest.raises(refresh_module.EvidenceRefreshError, match="single line"):
        refresh_module._validated_report_date("3 de agosto\ninyectado")
    with pytest.raises(refresh_module.EvidenceRefreshError, match="identifier contract"):
        refresh_module._required_run_id("not-a-run-id", "a" * 64)
