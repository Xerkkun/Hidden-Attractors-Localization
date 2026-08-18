"""Focused tests for the standalone MAVPD completed-run validator."""

from __future__ import annotations

import csv
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "python" / "validate_mavpd_integer_hidden_chaos_run.py"


def _module():
    spec = importlib.util.spec_from_file_location("validate_mavpd_completed_run", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _cubic_roots(a1: float, a2: float, a3: float) -> tuple[complex, complex, complex]:
    """Solve x^3+a1*x^2+a2*x+a3 with the complex Cardano formula."""

    p_value = a2 - a1 * a1 / 3.0
    q_value = 2.0 * a1**3 / 27.0 - a1 * a2 / 3.0 + a3
    discriminant = complex((q_value / 2.0) ** 2 + (p_value / 3.0) ** 3, 0.0)
    first = -q_value / 2.0 + discriminant**0.5
    second = -q_value / 2.0 - discriminant**0.5
    u_value = first ** (1.0 / 3.0) if abs(first) >= abs(second) else second ** (1.0 / 3.0)
    if abs(u_value) <= 1.0e-15:
        u_value = complex(-q_value, 0.0) ** (1.0 / 3.0)
    v_value = -p_value / (3.0 * u_value) if abs(u_value) > 1.0e-15 else 0j
    root_of_unity = complex(-0.5, math.sqrt(3.0) / 2.0)
    shift = a1 / 3.0
    return (
        u_value + v_value - shift,
        root_of_unity * u_value + root_of_unity.conjugate() * v_value - shift,
        root_of_unity.conjugate() * u_value + root_of_unity * v_value - shift,
    )


def _rebind(run_dir: Path, *relative_paths: str) -> None:
    status_path = run_dir / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    for relative in relative_paths:
        status["artifacts"][relative] = _digest(run_dir / relative)
    _write_json(status_path, status)


def _tree_state(root: Path) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(root).as_posix(): (_digest(path), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _summary_family(
    count: int,
    equilibria: list[str],
    *,
    declared: list[str] | None = None,
) -> dict:
    declared_equilibria = equilibria if declared is None else declared
    per_equilibrium = count // len(equilibria)
    return {
        "n_probes": count,
        "target_hits": 0,
        "ambiguous": 0,
        "numerical_failures": 0,
        "sampled_hiddenness_status": "hidden_under_tested_neighborhoods",
        "finite_sample_only": True,
        "global_hiddenness_proved": False,
        "tested_equilibria": equilibria,
        "required_equilibria": equilibria,
        "declared_equilibria": declared_equilibria,
        "tested_all_required_equilibria": True,
        "tested_all_declared_equilibria": all(value in equilibria for value in declared_equilibria),
        "by_equilibrium": {
            equilibrium: {
                "n": per_equilibrium,
                "target_hits": 0,
                "ambiguous": 0,
                "numerical_failures": 0,
                "equilibrium_destinations": per_equilibrium,
            }
            for equilibrium in equilibria
        },
    }


def _coverage(equilibria: list[str], radii: list[float], directions: int) -> dict:
    return {
        "complete": True,
        "expected_cells": len(equilibria) * len(radii),
        "expected_per_cell": directions,
        "cells": [
            {
                "equilibrium": equilibrium,
                "radius": radius,
                "expected": directions,
                "recorded": directions,
                "completed": directions,
                "unique_direction_ids": directions,
                "complete": True,
            }
            for equilibrium in equilibria
            for radius in radii
        ],
    }


def _build_legacy_minimal_run(tmp_path: Path):
    module = _module()
    repo = tmp_path / "repo"
    run_dir = repo / "tmp" / "completed-full-run"
    run_dir.mkdir(parents=True)

    for relative in module.SCIENTIFIC_SOURCE_FIXED_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{relative}\n", encoding="utf-8")
    package = repo / "hidden_attractors"
    package.mkdir()
    (package / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    (package / "solver.py").write_text("METHOD = 'test'\n", encoding="utf-8")
    source = module._scientific_source_snapshot(repo)

    run_id = (
        "mavpd_integer_hidden_chaos-full-20260803T120000000000Z-"
        f"{source['bundle_sha256'][:12]}-0123456789ab"
    )
    config_sha = "a" * 64
    contract = {
        "case_id": module.CASE_ID,
        "frequency_grid_used_for_seed": False,
        "fallback_frequency_scan_used": False,
        "direct_seed_records": [{"frequency_grid_used": False}],
        "scientific_source_snapshot": source,
    }
    selection = {
        "no_frequency_sweep": True,
        "scientific_source_snapshot": source,
    }

    probe_rows: list[dict] = []
    initial_rows: list[dict] = []
    specifications = (
        ("main_3x3xN", ["E0", "E+", "E-"], [1.0e-5, 1.0e-3, 1.0e-2], 12),
        ("targeted_E0_unstable", ["E0"], [1.0e-7, 1.0e-4], 2),
    )
    for family, equilibria, radii, directions in specifications:
        sample_id = 0
        for equilibrium in equilibria:
            for radius in radii:
                for direction in range(1, directions + 1):
                    coordinates = [float(sample_id), float(radius), float(direction)]
                    identity = {
                        "contract": family,
                        "sample_id": sample_id,
                        "equilibrium": equilibrium,
                        "radius": radius,
                        "direction_id": direction,
                    }
                    probe_rows.append(
                        {
                            **identity,
                            "sampling_mode": "sphere",
                            "x0": json.dumps(coordinates, separators=(",", ":")),
                            "status": "ok",
                            "destination": f"equilibrium_{equilibrium}",
                            "target_classification": "different_from_target_under_calibrated_cloud_test",
                            "target_distance_norm": 1.0,
                            "target_hit": False,
                            "ambiguous": False,
                            "tail_span": 0.0,
                            "closest_equilibrium": equilibrium,
                            "closest_equilibrium_distance": 0.0,
                        }
                    )
                    initial_rows.append(
                        {
                            **identity,
                            "y1": coordinates[0],
                            "y2": coordinates[1],
                            "y3": coordinates[2],
                        }
                    )
                    sample_id += 1
    _write_csv(run_dir / "07_hiddenness_probes.csv", probe_rows)
    _write_csv(run_dir / "07_hiddenness_initial_conditions.csv", initial_rows)

    hidden_summary = {
        "sampled_hiddenness_status": "hidden_under_tested_neighborhoods",
        "target_hits": 0,
        "ambiguous": 0,
        "numerical_failures": 0,
        "main": _summary_family(108, ["E0", "E+", "E-"]),
        "targeted_E0_unstable_direction": _summary_family(4, ["E0"]),
        "coverage_by_equilibrium_radius": {
            "complete": True,
            "main": _coverage(["E0", "E+", "E-"], [1.0e-5, 1.0e-3, 1.0e-2], 12),
            "targeted_E0_unstable_direction": _coverage(["E0"], [1.0e-7, 1.0e-4], 2),
        },
    }
    gate = {
        "chaotic_hidden_promotion_allowed": True,
        "hidden_chaos_status": "chaotic_hidden_under_tested_neighborhoods",
        "verdict": "hidden_under_tested_neighborhoods",
        "missing_conditions": [],
        "warnings": [],
        "diagnostic_conflicts": [],
    }
    gate_file = {
        "gate": gate,
        "evidence": {
            "run_metadata": {
                "run_id": run_id,
                "provenance": {
                    "scientific_source_snapshot": source,
                    "frequency_grid_used_for_search": False,
                },
            }
        },
    }

    figures_dir = run_dir / "figures"
    figures_dir.mkdir()
    global_root = repo / "outputs" / "library_figures"
    local_figure_rows = []
    global_figure_rows = []
    for figure_id in module.FIGURE_IDS:
        metadata = {
            "caption_key": f"fig_{figure_id}",
            "source_script": module.SCIENTIFIC_SOURCE_FIXED_FILES[0],
            "source_function": "test",
            "data_sources": [],
            "system_id": "modified_van_der_pol_duffing",
            "q": 1.0,
            "parameters": {"gamma": 0.1538},
            "integrator": "DOP853",
            "memory_mode": "not_applicable_integer_q1",
            "t_final": 900.0,
            "t_burn": 300.0,
            "scientific_source_bundle_sha256": source["bundle_sha256"],
            "quick_smoke_only": False,
        }
        local_png = figures_dir / f"{figure_id}.png"
        local_pdf = figures_dir / f"{figure_id}.pdf"
        local_png.write_bytes(f"png:{figure_id}".encode("utf-8"))
        local_pdf.write_bytes(f"pdf:{figure_id}".encode("utf-8"))
        central_paths = {}
        for suffix, local_path in (("png", local_png), ("pdf", local_pdf)):
            for target in (
                global_root / "by_run" / run_id / suffix / f"{figure_id}.{suffix}",
                global_root / "current" / suffix / f"{figure_id}.{suffix}",
                global_root
                / "by_export"
                / "mavpd_integer_hidden_chaos_report"
                / suffix
                / f"{figure_id}.{suffix}",
            ):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(local_path.read_bytes())
            central_paths[suffix] = str(
                global_root / "by_run" / run_id / suffix / f"{figure_id}.{suffix}"
            )
        metadata_path = global_root / "by_run" / run_id / "metadata" / f"{figure_id}.json"
        _write_json(metadata_path, metadata)
        local_figure_rows.append(
            {
                "figure_id": figure_id,
                "run_id": run_id,
                "local_png": f"figures/{figure_id}.png",
                "local_pdf": f"figures/{figure_id}.pdf",
                "png_sha256": _digest(local_png),
                "pdf_sha256": _digest(local_pdf),
                "global_promotion_requested": True,
                "promoted_to_global_manifest": True,
                "central_paths": central_paths,
                "metadata": metadata,
            }
        )
        global_figure_rows.append(
            {
                "figure_id": figure_id,
                "run_id": run_id,
                "kind": "mavpd_integer_hidden_chaos",
                "export_targets": ["mavpd_integer_hidden_chaos_report"],
                "pdf_path": f"library_figures/by_run/{run_id}/pdf/{figure_id}.pdf",
                "png_path": f"library_figures/by_run/{run_id}/png/{figure_id}.png",
                "metadata_path": f"library_figures/by_run/{run_id}/metadata/{figure_id}.json",
            }
        )

    global_manifest_json = global_root / "manifests" / "figure_manifest.json"
    global_manifest_csv = global_root / "manifests" / "figure_manifest.csv"
    _write_json(global_manifest_json, global_figure_rows)
    _write_csv(
        global_manifest_csv,
        [
            {
                **row,
                "export_targets": json.dumps(row["export_targets"]),
            }
            for row in global_figure_rows
        ],
    )
    receipt = {
        "status": "committed",
        "run_id": run_id,
        "scientific_source_bundle_sha256": source["bundle_sha256"],
        "figure_count": 8,
        "figure_ids": list(module.FIGURE_IDS),
        "seconds": 0.25,
        "global_manifest_paths": [str(global_manifest_json), str(global_manifest_csv)],
    }
    timings = [
        {"phase": "contract", "seconds": 1.0, "timing_source": "perf_counter"},
        {"phase": "search", "seconds": 2.0, "timing_source": "perf_counter"},
        {"phase": "diagnostics", "seconds": 2.0, "timing_source": "perf_counter"},
        {"phase": "hiddenness", "seconds": 2.0, "timing_source": "perf_counter"},
        {"phase": "candidate_gate", "seconds": 1.25, "timing_source": "perf_counter"},
        {"phase": "total", "seconds": 10.0, "timing_source": "perf_counter"},
    ]

    _write_json(run_dir / "00_system_contract.json", contract)
    _write_json(run_dir / "03_candidate_selection.json", selection)
    _write_json(run_dir / "07_hiddenness_summary.json", hidden_summary)
    _write_json(run_dir / "09_candidate_gate.json", gate_file)
    _write_json(figures_dir / "figure_manifest.json", {"figures": local_figure_rows})
    _write_json(figures_dir / "global_promotion_receipt.json", receipt)
    _write_csv(run_dir / "phase_timings.csv", timings)
    manifest = {
        "case_id": module.CASE_ID,
        "run_id": run_id,
        "config_sha256": config_sha,
        "quick_mode": False,
        "scientific_source_snapshot": source,
        "candidate_gate": gate,
        "hiddenness": hidden_summary,
        "figures": local_figure_rows,
        "frequency_grid_used_for_search": False,
        "global_proof_claimed": False,
        "global_figure_promotion": receipt,
        "timings": timings,
    }
    _write_json(run_dir / "run_manifest.json", manifest)

    for relative in module.REQUIRED_LEDGER_ARTIFACTS:
        path = run_dir / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            _write_json(path, {})
        elif path.suffix == ".csv":
            path.write_text("placeholder\n", encoding="utf-8")
        else:  # every binary figure was already created above
            raise AssertionError(f"test fixture did not create {relative}")
    ledger = {
        relative: _digest(run_dir / relative)
        for relative in sorted(module.REQUIRED_LEDGER_ARTIFACTS)
    }
    status = {
        "status": "complete",
        "run_id": run_id,
        "created_at_utc": "2026-08-03T12:00:00+00:00",
        "completed_at_utc": "2026-08-03T12:10:00+00:00",
        "quick_mode": False,
        "config_sha256": config_sha,
        "runtime_environment": {"python_version": "test"},
        "scientific_source_snapshot": source,
        "completed_phases": list(module.DIRECT_PHASES),
        "last_completed_phase": "global_figure_promotion",
        "artifacts": ledger,
    }
    _write_json(run_dir / "run_status.json", status)
    return module, repo, run_dir


def _build_valid_run(tmp_path: Path, *, figure_store_root: Path | None = None):
    """Build a compact but semantically complete full-run artifact graph."""

    module = _module()
    repo = tmp_path / "repo"
    run_dir = repo / "tmp" / "completed-full-run"
    run_dir.mkdir(parents=True)
    for relative in module.SCIENTIFIC_SOURCE_FIXED_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{relative}\n", encoding="utf-8")
    package = repo / "hidden_attractors"
    package.mkdir()
    (package / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    (package / "solver.py").write_text("METHOD = 'test'\n", encoding="utf-8")
    source = module._scientific_source_snapshot(repo)
    run_id = (
        "mavpd_integer_hidden_chaos-full-20260803T120000000000Z-"
        f"{source['bundle_sha256'][:12]}-0123456789ab"
    )
    config_sha = "a" * 64
    runtime = {
        "python_version": "3.test",
        "python_implementation": "CPython",
        "platform": "test-platform",
        "numpy_version": "test",
        "scipy_version": "test",
    }
    hopf = module._mavpd_high_hopf_boundary()
    selected_offset = 0.010
    selected_gamma = hopf + selected_offset
    candidate_parameters = {"delta": 100.0, "gamma": selected_gamma, "rho": 200.0, "xi": 2.85}

    def complex_value(real: float, imag: float = 0.0) -> dict:
        return {"real": real, "imag": imag}

    seed_rows = []
    transfer_rows = []
    for branch, (omega, gain) in enumerate(module._mavpd_direct_seed_pairs()):
        amplitude = (gain / 0.75) ** 0.5
        linear_y1 = 10.0 - 100.0 * gain
        eigenvector = (
            complex(1.0, 0.0),
            (1j * omega - linear_y1) / 100.0,
            0j,
        )
        eigenvector = (eigenvector[0], eigenvector[1], 200.0 * eigenvector[1] / (1j * omega))
        response = module._mavpd_transfer_response(omega)
        seed_rows.append(
            {
                "branch_index": branch,
                "phase": 0.0,
                "omega0": omega,
                "k": gain,
                "a0": amplitude,
                "seed": [amplitude * value.real for value in eigenvector],
                "eigenvector": [complex_value(value.real, value.imag) for value in eigenvector],
                "matched_eigenvalue": complex_value(0.0, omega),
                "method": "classic",
                "frequency_grid_used": False,
                "published_table_used": False,
                "search_route": "direct_integer_transfer",
            }
        )
        transfer_rows.append(
            {
                "branch_index": branch,
                "omega0": omega,
                "W_iomega": complex_value(response.real, response.imag),
                "imaginary_residual": abs(response.imag),
                "closure_residual": abs(response.real + 1.0 / gain),
                "describing_function_residual": abs(0.75 * amplitude * amplitude - gain),
            }
        )
    base_a = 0.1**0.5
    contract = {
        "case_id": module.CASE_ID,
        "source_doi": "10.3390/math11030591",
        "source_scope": "published_model_equations_only",
        "candidate_parameter_set_published": False,
        "scientific_source_snapshot": source,
        "equations": list(module.EXPECTED_EQUATIONS),
        "q": 1.0,
        "base_parameters": dict(module.EXPECTED_BASE_PARAMETERS),
        "lure": {
            "A": [[10.0, 100.0, 0.0], [1.0, -3.1, -1.0], [0.0, 200.0, 0.0]],
            "b": [-100.0, 0.0, 0.0],
            "c": [1.0, 0.0, 0.0],
            "psi": "sigma^3",
            "describing_function": "N(a)=3*a^2/4",
            "max_field_residual": 0.0,
        },
        "jacobian_residual": 0.0,
        "equilibria": [
            {
                "name": "E0",
                "state": [0.0, 0.0, 0.0],
                "rhs_residual": 0.0,
                "eigenvalues": [
                    complex_value(13.179182140318382),
                    complex_value(-3.1395910701591925, 11.91207173549869),
                    complex_value(-3.1395910701591925, -11.91207173549869),
                ],
            },
            {
                "name": "E+",
                "state": [base_a, 0.0, base_a],
                "rhs_residual": 0.0,
                "eigenvalues": [
                    complex_value(-23.461845785791205),
                    complex_value(0.18092289289559804, 13.055911953260793),
                    complex_value(0.18092289289559804, -13.055911953260793),
                ],
            },
            {
                "name": "E-",
                "state": [-base_a, 0.0, -base_a],
                "rhs_residual": 0.0,
                "eigenvalues": [
                    complex_value(-23.461845785791205),
                    complex_value(0.18092289289559804, 13.055911953260793),
                    complex_value(0.18092289289559804, -13.055911953260793),
                ],
            },
        ],
        "direct_seed_records": seed_rows,
        "transfer_checks": transfer_rows,
        "hopf_boundary_at_xi_target": {
            "gamma_boundaries": [0.017384798091736965, hopf],
            "selected_high_gamma_boundary": hopf,
            "routh_hurwitz_residual_at_boundary": 0.0,
            "values_derived_from_equations": True,
        },
        "frequency_grid_used_for_seed": False,
        "fallback_frequency_scan_used": False,
        "report_values_used_as_search_input": False,
        "mathematica_validation": "validation/wolfram/cases/mavpd_integer.wl",
    }
    _write_json(run_dir / "00_system_contract.json", contract)

    strict_branch_payload = {
        "method": "integer_dop853_variational_qr",
        "status": "ok",
        "exponents": [-0.1, -1.0, -10.0],
        "sum_exponents": -11.1,
        "final_state": [0.0, 0.0, 0.0],
        "t_accumulate": 250.0,
        "metadata": {
            "solver_method": "DOP853",
            "solver": "scipy.integrate.solve_ivp",
            "jacobian_source": "analytic",
            "jacobian_eps": None,
            "dimension": 3,
            "div_threshold": 50.0,
            "max_step": 0.02,
            "rtol": 2.0e-10,
            "atol": 2.0e-12,
            "qr_interval": 0.5,
            "qr_segments": 500,
            "t_burn_requested": 100.0,
            "t_burn_completed": 100.0,
            "t_accumulate_requested": 250.0,
            "t_accumulate_completed": 250.0,
        },
        "finite_time_local": True,
        "does_not_prove_chaos_alone": True,
    }
    branch_final_states = {0: [0.01, 0.02, 0.03], 1: [0.04, 0.05, 0.06]}
    route = {
        "primary_route": "direct_integer_lure",
        "frequency_grid_used": False,
        "branches": [
            {
                "branch_index": branch,
                "omega0": omega,
                "lambda_nodes_completed": 21,
                "final_lambda": 1.0,
                "final_state": branch_final_states[branch],
                "lyapunov": strict_branch_payload,
                "chaotic_at_base": False,
            }
            for branch, (omega, _gain) in enumerate(module._mavpd_direct_seed_pairs())
        ],
        "alternative_triggered": True,
        "trigger": "neither direct base branch passes the declared finite-time LLE chaos screen",
        "alternative_source_branch_index": 0,
        "alternative_source_branch_selection_rule": "lowest direct harmonic frequency among successful direct base branches",
    }
    _write_json(run_dir / "01_direct_seed_and_lambda_continuation.json", route)

    continuation_rows: list[dict] = []
    previous = branch_final_states[0]
    for index in range(25):
        xi = 3.09 - 0.01 * index
        x_out = [0.02 + index * 0.001, 0.01, -0.02]
        continuation_rows.append(
            {
                "stage": "xi",
                "node_index": index,
                "xi": xi,
                "gamma": 0.1,
                "x_in": json.dumps(previous),
                "x_out": json.dumps(x_out),
                "status": "ok",
                "system_rebuilt_at_node": True,
                "lure_rebuilt_at_node": True,
            }
        )
        previous = x_out
    selected_state = None
    gamma_states: dict[float, list[float]] = {}
    largest_gamma = hopf + max(module.EXPECTED_SCREEN_OFFSETS)
    base_gamma_nodes: list[float] = []
    node = 1
    while 0.1 + 0.0025 * node < largest_gamma - 1.0e-14:
        base_gamma_nodes.append(0.1 + 0.0025 * node)
        node += 1
    gamma_nodes = sorted(base_gamma_nodes + [hopf + offset for offset in module.EXPECTED_SCREEN_OFFSETS])
    for index, gamma in enumerate(gamma_nodes):
        x_out = [gamma, 0.02, -0.2]
        matching_offset = next(
            (offset for offset in module.EXPECTED_SCREEN_OFFSETS if math.isclose(gamma, hopf + offset, rel_tol=0.0, abs_tol=1.0e-12)),
            None,
        )
        if matching_offset is not None:
            gamma_states[matching_offset] = x_out
        continuation_rows.append(
            {
                "stage": "gamma",
                "node_index": index,
                "xi": 2.85,
                "gamma": gamma,
                "x_in": json.dumps(previous),
                "x_out": json.dumps(x_out),
                "status": "ok",
                "system_rebuilt_at_node": True,
                "lure_rebuilt_at_node": True,
            }
        )
        previous = x_out
        if matching_offset == selected_offset:
            selected_state = x_out
    assert selected_state is not None
    _write_csv(run_dir / "02_parameter_continuation.csv", continuation_rows)

    screen_rows: list[dict] = []
    screen_probe_rows: list[dict] = []
    six_directions = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
    )
    for offset in module.EXPECTED_SCREEN_OFFSETS:
        gamma = hopf + offset
        eligible = offset <= selected_offset
        screen_rows.append(
            {
                "hopf_offset": offset,
                "gamma": gamma,
                "equilibrium_stability_margin": 0.05,
                "lambda_1": 0.8 if eligible else 0.1,
                "lambda_2": 0.0,
                "lambda_3": -20.0,
                "lyapunov_status": "ok",
                "E0_probe_count": 12,
                "E0_target_hits": 0,
                "E0_ambiguous": 0,
                "eligible_hidden_chaos_screen": eligible,
            }
        )
        sample_id = 0
        for radius in (1.0e-7, 1.0e-4):
            for direction_id, direction in enumerate(six_directions, start=1):
                x0 = [radius * value for value in direction]
                screen_probe_rows.append(
                    {
                        "hopf_offset": offset,
                        "gamma": gamma,
                        "reference_acceptance_threshold": 0.01,
                        "contract": "candidate_screen_E0",
                        "sample_id": sample_id,
                        "equilibrium": "E0",
                        "radius": radius,
                        "direction_id": direction_id,
                        "sampling_mode": "sphere",
                        "x0": json.dumps(x0),
                        "status": "ok",
                        "destination": "different_from_target_under_calibrated_cloud_test",
                        "target_classification": "different_from_target_under_calibrated_cloud_test",
                        "target_distance_norm": 1.0,
                        "target_hit": False,
                        "ambiguous": False,
                        "tail_span": 0.1,
                        "closest_equilibrium": "E0",
                        "closest_equilibrium_distance": 0.1,
                    }
                )
                sample_id += 1
    _write_csv(run_dir / "03_candidate_screening.csv", screen_rows)
    _write_csv(run_dir / "03_candidate_screening_probes.csv", screen_probe_rows)
    _write_json(
        run_dir / "03_candidate_screening_contract.json",
        {
            "contract": {
                "lyapunov": {
                    "t_burn": 100.0,
                    "t_accumulate": 250.0,
                    "qr_interval": 0.5,
                    "rtol": 2.0e-10,
                    "atol": 2.0e-12,
                    "max_step": 0.02,
                    "positive_threshold": 0.5,
                },
                "reference": {
                    "duration": 360.0,
                    "burn": 160.0,
                    "sample_step": 0.05,
                    "max_step": 0.03,
                    "safety_factor": 3.0,
                    "max_points": 600,
                },
                "probes": {
                    "equilibrium_names": ["E0"],
                    "radii": [1.0e-7, 1.0e-4],
                    "directions": 6,
                    "sampling_mode": "sphere",
                    "t_burn": 500.0,
                    "t_keep": 100.0,
                    "sample_step": 0.05,
                    "rtol": 2.0e-10,
                    "atol": 2.0e-12,
                    "max_step": 0.03,
                    "equilibrium_tol": 1.0e-6,
                    "equilibrium_tail_span_tol": 1.0e-5,
                },
            },
            "candidate_states_source": "02_parameter_continuation.csv x_out at each exact Hopf-offset node",
            "probe_initial_conditions": "03_candidate_screening_probes.csv",
            "finite_sample_only": True,
        },
    )
    selected_screen = next(row for row in screen_rows if row["hopf_offset"] == selected_offset)
    selection = {
        "hopf_boundary": hopf,
        "selection_rule": module.EXPECTED_SELECTION_RULE,
        "selected": selected_screen,
        "selected_candidate_initial_state": selected_state,
        "parameter_provenance": "gamma_selected_by_local_continuation_at_declared_xi_endpoint_not_a_published_parameter_tuple",
        "xi_endpoint_provenance": "declared_local_continuation_endpoint_not_selected_by_candidate_screen",
        "alternative_source_branch_index": 0,
        "alternative_source_branch_selection_rule": "lowest direct harmonic frequency among successful direct base branches",
        "scientific_source_snapshot": source,
        "no_frequency_sweep": True,
        "primary_route_chaos_screen_failed_before_alternative": True,
    }
    _write_json(run_dir / "03_candidate_selection.json", selection)

    trajectory_rows: list[dict] = []
    for index in range(45001):
        time_value = 0.02 * index
        if index == 0:
            y1, y2, y3 = selected_state
        else:
            y1 = 0.4 + 0.1 * math.sin(0.11 * time_value)
            y2 = 0.25 * math.sin(2.0 * time_value)
            y3 = 0.1 * math.sin(0.07 * time_value + 0.2)
        trajectory_rows.append({"time": time_value, "y1": y1, "y2": y2, "y3": y3})
    _write_csv(run_dir / "04_candidate_trajectory.csv", trajectory_rows)
    post_states = [row for row in trajectory_rows if row["time"] >= 300.0]
    post_count = len(post_states)
    norms = [(row["y1"] ** 2 + row["y2"] ** 2 + row["y3"] ** 2) ** 0.5 for row in post_states]
    max_norm = max(norms)
    sorted_norms = sorted(norms)
    median_norm = sorted_norms[len(sorted_norms) // 2]
    window = max(2, round(len(norms) * 0.2))
    early_mean = sum(norms[:window]) / window
    late_mean = sum(norms[-window:]) / window
    coordinate_min = [min(row[field] for row in post_states) for field in ("y1", "y2", "y3")]
    coordinate_max = [max(row[field] for row in post_states) for field in ("y1", "y2", "y3")]
    coordinate_span = [upper - lower for lower, upper in zip(coordinate_min, coordinate_max)]
    mean_divergence = sum(
        selected_gamma * 100.0 - 2.85 - 300.0 * row["y1"] ** 2
        for row in post_states
    ) / post_count

    strict_exponents = [0.7, 0.0, mean_divergence - 0.7]
    control_exponents = [0.68, 0.0, mean_divergence - 0.68]
    strict_meta = {
        "solver_method": "DOP853",
        "solver": "scipy.integrate.solve_ivp",
        "jacobian_source": "analytic",
        "jacobian_eps": None,
        "dimension": 3,
        "div_threshold": 50.0,
        "max_step": 0.01,
        "rtol": 2.0e-12,
        "atol": 2.0e-14,
        "qr_interval": 0.5,
        "qr_segments": 2400,
        "t_burn_requested": 300.0,
        "t_burn_completed": 300.0,
        "t_accumulate_requested": 1200.0,
        "t_accumulate_completed": 1200.0,
    }
    control_meta = {
        "solver_method": "DOP853",
        "solver": "scipy.integrate.solve_ivp",
        "jacobian_source": "analytic",
        "jacobian_eps": None,
        "dimension": 3,
        "div_threshold": 50.0,
        "max_step": 0.02,
        "rtol": 2.0e-10,
        "atol": 2.0e-12,
        "qr_interval": 0.5,
        "qr_segments": 1200,
        "t_burn_requested": 200.0,
        "t_burn_completed": 200.0,
        "t_accumulate_requested": 600.0,
        "t_accumulate_completed": 600.0,
    }
    def le_payload(values: list[float], metadata: dict) -> dict:
        return {
            "method": "integer_dop853_variational_qr",
            "status": "ok",
            "exponents": values,
            "sum_exponents": sum(values),
            "final_state": [0.1, 0.2, 0.3],
            "t_accumulate": metadata["t_accumulate_completed"],
            "metadata": metadata,
            "finite_time_local": True,
            "does_not_prove_chaos_alone": True,
        }
    return_map = [
        {
            "coordinate": coordinate,
            "K": k_value,
            "K_median": k_value,
            "K_mean": k_value,
            "K_min": k_value - 0.01,
            "K_max": k_value + 0.005,
            "K_std": 0.005,
            "c_values_count": 64,
            "signal_length": 120,
            "random_seed": 20260802,
            "state": "zero_one_chaotic_candidate",
            "detrend": True,
            "normalize": True,
            "zero_one_alone_does_not_certify_chaos": True,
            "chaos_certified_by_zero_one": False,
            "hiddenness_certified_by_zero_one": False,
        }
        for coordinate, k_value in ((0, 0.99), (2, 0.98))
    ]
    poincare_rows = []
    previous_crossing_time = None
    for left, right in zip(trajectory_rows, trajectory_rows[1:]):
        left_y2 = left["y2"]
        right_y2 = right["y2"]
        delta_y2 = right_y2 - left_y2
        if not (left_y2 < 0.0 <= right_y2 and delta_y2 > 0.0):
            continue
        theta = -left_y2 / delta_y2
        crossing_time = left["time"] + theta * (right["time"] - left["time"])
        if crossing_time < 300.0:
            continue
        y1_crossing = left["y1"] + theta * (right["y1"] - left["y1"])
        y3_crossing = left["y3"] + theta * (right["y3"] - left["y3"])
        if y1_crossing - y3_crossing <= 0.0:
            continue
        if previous_crossing_time is not None and crossing_time - previous_crossing_time < 0.05:
            continue
        poincare_rows.append({"time": crossing_time, "y1": y1_crossing, "y3": y3_crossing})
        previous_crossing_time = crossing_time
    for row in return_map:
        row["signal_length"] = len(poincare_rows)
    poincare_points = [(row["y1"], row["y3"]) for row in poincare_rows]
    poincare_centroid = [
        sum(point[index] for point in poincare_points) / len(poincare_points)
        for index in (0, 1)
    ]
    poincare_covariance = [
        [
            sum(
                (point[i] - poincare_centroid[i]) * (point[j] - poincare_centroid[j])
                for point in poincare_points
            )
            / (len(poincare_points) - 1)
            for j in (0, 1)
        ]
        for i in (0, 1)
    ]
    poincare_unique = len(
        {(round(point[0] / 1.0e-6), round(point[1] / 1.0e-6)) for point in poincare_points}
    )
    poincare_nearest = [
        min(
            math.hypot(point[0] - other[0], point[1] - other[1])
            for other_index, other in enumerate(poincare_points)
            if other_index != point_index
        )
        for point_index, point in enumerate(poincare_points)
    ]
    poincare_nearest_median = module._median(poincare_nearest)
    poincare_summary = {
        "crossing_count": len(poincare_rows),
        "retained_after_burn": len(poincare_rows),
        "bounding_box": {
            f"coordinate_{index}": {
                "minimum": min(point[index] for point in poincare_points),
                "maximum": max(point[index] for point in poincare_points),
            }
            for index in (0, 1)
        },
        "centroid": poincare_centroid,
        "covariance": poincare_covariance,
        "rank_estimate": 2,
        "duplicate_fraction": 1.0 - poincare_unique / len(poincare_points),
        "nearest_neighbor_stats": {
            "minimum": min(poincare_nearest),
            "median": poincare_nearest_median,
            "mean": sum(poincare_nearest) / len(poincare_nearest),
            "maximum": max(poincare_nearest),
        },
        "interpretation_label": "cloud_like",
        "section_metadata": {
            "section_variable": 1,
            "section_index": 1,
            "section_value": 0.0,
            "direction": "positive",
            "direction_rule": "rhs(section_crossing)[section_variable] has requested sign",
            "derivative_mode": "integer_rhs",
            "interpolation": "linear",
            "min_crossing_separation": 0.05,
            "filtered_by_min_crossing_separation": 0,
            "burn_time": 300.0,
            "caputo_geometric_crossing": False,
            "exact_poincare_map": False,
            "sampled_linear_interpolation": True,
            "classical_integer_section_interpretation": True,
            "uses_classical_rhs_direction": True,
        },
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
    stride_rows = []
    for stride in (5, 10, 15, 20, 25, 30, 40, 50, 75, 100):
        stride_rows.append(
            {
                "series": "flow_y1",
                "stride": stride,
                "effective_sample_step": 0.02 * stride,
                "samples": (post_count + stride - 1) // stride,
                "K": 0.5,
                "state": "zero_one_inconclusive",
            }
        )
    diagnostics = {
        "candidate_parameters": candidate_parameters,
        "boundedness": {
            "boundedness_status": "bounded_candidate",
            "finite_fraction": 1.0,
            "nonfinite_count": 0,
            "post_transient_rows": post_count,
            "max_norm": max_norm,
            "R_observed": max_norm,
            "min_norm": min(norms),
            "mean_norm": sum(norms) / len(norms),
            "median_norm": median_norm,
            "final_norm": norms[-1],
            "norm_growth_ratio": late_mean / early_mean,
            "burn_time": 300.0,
            "norm": "euclidean",
            "divergence_radius": 50.0,
            "coordinate_min": coordinate_min,
            "coordinate_max": coordinate_max,
            "coordinate_span": coordinate_span,
            "boundedness_proves_chaos": False,
            "chaos_certified_by_boundedness": False,
            "hiddenness_certified_by_boundedness": False,
        },
        "reference_calibration": {
            "status": "calibrated",
            "acceptance_threshold": 0.0087,
            "ambiguity_margin": 0.0029,
            "scale": 1.0,
            "max_points": 1000,
            "within_reference_distances": [0.001, 0.002, 0.003],
            "negative_control_distances": [0.1, 0.2, 0.3],
        },
        "lyapunov": le_payload(strict_exponents, strict_meta),
        "lyapunov_control": le_payload(control_exponents, control_meta),
        "kaplan_yorke_dimension": 2.0 + 0.7 / abs(strict_exponents[2]),
        "mean_vector_field_divergence": mean_divergence,
        "lyapunov_sum_minus_mean_divergence": sum(strict_exponents) - mean_divergence,
        "zero_one": {"flow_stride_sensitivity": stride_rows, "return_map_results": return_map},
        "poincare": poincare_summary,
        "spectrum": {
            "state_global": "spectral_inconclusive",
            "coordinate_results": {},
            "gate_applicable": False,
            "normalized_fft_power_not_welch_psd": True,
        },
        "efork_crosscheck": {
            "status": "ok",
            "h": 0.002,
            "t_final": 600.0,
            "tail_burn_fraction": 0.5,
            "target_match": {
                "classification": "same_attractor_under_calibrated_cloud_test",
                "finite_sample_only": True,
                "distance_norm": 0.002,
                "distances_norm": [0.001, 0.002, 0.003],
                "acceptance_threshold": 0.0087,
                "ambiguity_upper_bound": 0.0116,
                "calibration_status": "calibrated",
            },
        },
        "finite_time_only": True,
    }
    _write_json(run_dir / "05_chaos_diagnostics.json", diagnostics)
    _write_csv(
        run_dir / "05_lyapunov_convergence.csv",
        [
            {
                "time": 0.5 * index,
                "lambda_1": strict_exponents[0] if index == 2400 else 0.65 + 0.05 * index / 2400.0,
                "lambda_2": strict_exponents[1] if index == 2400 else 0.01 * (1.0 - index / 2400.0),
                "lambda_3": strict_exponents[2] if index == 2400 else -19.0 - index / 2400.0,
            }
            for index in range(1, 2401)
        ],
    )
    _write_csv(run_dir / "05_poincare_section.csv", poincare_rows)
    _write_json(run_dir / "05_zero_one_return_map.json", return_map)
    _write_csv(run_dir / "05_zero_one_stride_sensitivity.csv", stride_rows)
    spectrum_rows = [
        {"coordinate": coordinate, "frequency": frequency, "normalized_fft_power": power}
        for coordinate in ("x", "y", "z")
        for frequency, power in ((0.0, 0.1), (1.0, 0.9))
    ]
    _write_csv(run_dir / "05_normalized_fft_power.csv", spectrum_rows)

    candidate_a = selected_gamma**0.5
    stability = []
    candidate_spectra: dict[str, tuple[complex, complex, complex]] = {}
    for equilibrium, state, linear_y1 in (
        ("E0", [0.0, 0.0, 0.0], 100.0 * selected_gamma),
        ("E+", [candidate_a, 0.0, candidate_a], -200.0 * selected_gamma),
        ("E-", [-candidate_a, 0.0, -candidate_a], -200.0 * selected_gamma),
    ):
        coefficients = module._mavpd_characteristic_coefficients(
            linear_y1=linear_y1,
            delta=100.0,
            rho=200.0,
            xi=2.85,
        )
        roots = _cubic_roots(*coefficients)
        candidate_spectra[equilibrium] = roots
        spectral_abscissa = max(value.real for value in roots)
        stability.append(
            {
                "equilibrium": equilibrium,
                "state": state,
                "rhs_residual": 0.0,
                "eigenvalues": [complex_value(value.real, value.imag) for value in roots],
                "spectral_abscissa": spectral_abscissa,
                "stability": "unstable" if equilibrium == "E0" else "locally_asymptotically_stable",
            }
        )
    _write_json(run_dir / "06_equilibrium_stability.json", stability)
    _write_json(run_dir / "08_robustness_matrix.json", robustness)

    candidate_equilibria = {row["equilibrium"]: row["state"] for row in stability}
    unstable_eigenvalue = max(candidate_spectra["E0"], key=lambda value: value.real)
    unstable_y2 = (unstable_eigenvalue.real - 100.0 * selected_gamma) / 100.0
    unstable_y3 = 200.0 * unstable_y2 / unstable_eigenvalue.real
    unstable_direction = [1.0, unstable_y2, unstable_y3]
    unstable_norm = math.sqrt(sum(value * value for value in unstable_direction))
    unstable_direction = [value / unstable_norm for value in unstable_direction]
    main_directions = module._deterministic_unit_directions_3d(12)
    probe_rows: list[dict] = []
    initial_rows: list[dict] = []
    specifications = (
        ("main_3x3xN", ["E0", "E+", "E-"], [1.0e-5, 1.0e-3, 1.0e-2], main_directions),
        (
            "targeted_E0_unstable",
            ["E0"],
            [1.0e-7, 1.0e-4],
            (tuple(unstable_direction), tuple(-value for value in unstable_direction)),
        ),
    )
    for family, equilibria, radii, directions in specifications:
        sample_id = 0
        for equilibrium in equilibria:
            center = candidate_equilibria[equilibrium]
            for radius in radii:
                for direction_id, direction in enumerate(directions, start=1):
                    coordinates = [center[index] + radius * direction[index] for index in range(3)]
                    identity = {"contract": family, "sample_id": sample_id, "equilibrium": equilibrium, "radius": radius, "direction_id": direction_id}
                    probe_rows.append(
                        {
                            **identity,
                            "sampling_mode": "sphere",
                            "x0": json.dumps(coordinates, separators=(",", ":")),
                            "status": "ok",
                            "destination": f"equilibrium_{equilibrium}",
                            "target_classification": "different_from_target_under_calibrated_cloud_test",
                            "target_distance_norm": 1.0,
                            "target_hit": False,
                            "ambiguous": False,
                            "tail_span": 0.0,
                            "closest_equilibrium": equilibrium,
                            "closest_equilibrium_distance": 0.0,
                        }
                    )
                    initial_rows.append({**identity, "y1": coordinates[0], "y2": coordinates[1], "y3": coordinates[2]})
                    sample_id += 1
    _write_csv(run_dir / "07_hiddenness_probes.csv", probe_rows)
    _write_csv(run_dir / "07_hiddenness_initial_conditions.csv", initial_rows)
    main_summary = _summary_family(108, ["E0", "E+", "E-"], declared=["E0", "E+", "E-"])
    targeted_summary = _summary_family(4, ["E0"], declared=["E0", "E+", "E-"])
    targeted_summary.update(
        {
            "unstable_eigenvalue": complex_value(unstable_eigenvalue.real, unstable_eigenvalue.imag),
            "unstable_direction": unstable_direction,
        }
    )
    hidden_summary = {
        "sampled_hiddenness_status": "hidden_under_tested_neighborhoods",
        "n_probes": 112,
        "target_hits": 0,
        "ambiguous": 0,
        "numerical_failures": 0,
        "finite_sample_only": True,
        "global_hiddenness_proved": False,
        "main": main_summary,
        "targeted_E0_unstable_direction": targeted_summary,
        "coverage_by_equilibrium_radius": {
            "complete": True,
            "main": _coverage(["E0", "E+", "E-"], [1.0e-5, 1.0e-3, 1.0e-2], 12),
            "targeted_E0_unstable_direction": _coverage(["E0"], [1.0e-7, 1.0e-4], 2),
        },
    }
    _write_json(run_dir / "07_hiddenness_summary.json", hidden_summary)

    checked_conditions = {key: True for key in module.EXPECTED_GATE_CONDITIONS}
    eta_path = [index / 20.0 for index in range(21)]
    gate_tolerances = {
        "boundedness_norm": 120.0,
        "equilibrium_residual_tol": 1.0e-8,
        "lyapunov_positive_tol": 0.02,
        "matignon_tol": 1.0e-12,
        "nontrivial_variance_tol": 1.0e-8,
        "spectral_peak_dominance_threshold": 0.8,
        "target_match_tol": 0.5,
        "zero_one_chaos_threshold": 0.7,
        "zero_one_regular_threshold": 0.3,
    }
    run_metadata = {
        "schema_version": "1.0",
        "run_id": run_id,
        "workflow": "integer_hidden_chaos_search",
        "system": "modified-van-der-pol-duffing",
        "created_at_utc": "2026-08-03T12:00:00+00:00",
        "numerical_contract": {
            "q": 1.0,
            "h": 0.02,
            "t_final": 900.0,
            "t_burn": 300.0,
            "memory": {
                "mode": "not_applicable",
                "M": None,
                "memory_window_steps": None,
                "memory_window_time": None,
                "is_full_caputo": False,
            },
            "integrator": {"name": "DOP853", "backend": "python", "caputo": False},
        },
        "software": {
            "python_version": "3.test",
            "platform": "test-platform",
            "package_version": "1.2.0",
            "numpy_version": "test",
            "scipy_version": "test",
            "git_commit": "1" * 40,
            "working_tree_dirty": False,
        },
        "continuation": {
            "used": True,
            "eta_path": eta_path,
            "continuation_mode": "integer",
            "memory_window_propagated": None,
            "final_eta": 1.0,
        },
        "tolerances": gate_tolerances,
        "parameters": candidate_parameters,
        "lure": {
            "matrix": [[100.0 * selected_gamma, 100.0, 0.0], [1.0, -2.85, -1.0], [0.0, 200.0, 0.0]],
            "input_vector": [-100.0, 0.0, 0.0],
            "output_vector": [1.0, 0.0, 0.0],
            "scalar_nonlinearity": "mavpd_lure_system.<locals>.nonlinearity",
            "transfer_convention": "c^T (P-s I)^(-1) b; direct polynomial roots",
            "harmonic_condition": "Im W(i omega)=0 and Re W(i omega)=-1/k",
        },
        "seed": {
            "candidate_id": "mavpd-chaos-continuation-endpoint",
            "family": "continuation_endpoint_from_theoretical_lure_seed",
            "x0": selected_state,
            "source": "selected_parameter_continuation_endpoint",
            "parameters": {
                "source_branch_index": 0,
                "theoretical_harmonic_seed": seed_rows[0]["seed"],
                "omega0": seed_rows[0]["omega0"],
                "k": seed_rows[0]["k"],
                "a0": seed_rows[0]["a0"],
            },
        },
        "random_seed": 20260802,
        "random_seed_policy": "fixed_reproducible",
        "provenance": {
            "source_doi": "10.3390/math11030591",
            "source_scope": "published_model_equations_only",
            "candidate_parameter_set": "gamma_selected_at_declared_xi_continuation_endpoint",
            "candidate_parameter_set_published": False,
            "scientific_source_snapshot": source,
            "frequency_grid_used_for_search": False,
            "alternative_triggered_after_direct_chaos_screen_failure": True,
        },
        "extra": {"candidate_parameters": candidate_parameters},
    }
    gate = {
        "attractor_status": "hidden_under_tested_neighborhoods",
        "verdict": "hidden_under_tested_neighborhoods",
        "hiddenness_evidence_level": "hidden_under_tested_neighborhoods",
        "evidence_level": "hidden_under_tested_neighborhoods",
        "chaos_evidence_level": "strong_chaos_evidence",
        "lyapunov_support": "positive",
        "zero_one_support": "chaotic",
        "spectral_support": "not_available_or_inconclusive",
        "boundedness_support": "bounded_nontrivial",
        "poincare_support": "chaotic",
        "diagnostic_conflicts": [],
        "checked_conditions": checked_conditions,
        "missing_conditions": [],
        "warnings": [],
        "promotion_allowed": True,
        "hiddenness_promotion_allowed": True,
        "chaotic_hidden_promotion_allowed": True,
        "hidden_chaos_status": "chaotic_hidden_under_tested_neighborhoods",
    }
    gate_file = {
        "gate": gate,
        "evidence": {
            "run_metadata": run_metadata,
            "equilibria": {"all_found": True, "max_residual": 0.0},
            "matignon": {"all_classified": True, "q": 1.0},
            "seed": {
                "localized": True,
                "method": "continuation",
                "source": "candidate_initial_state_is_selected_parameter_continuation_endpoint",
                "theoretical_seed_source": "direct_integer_transfer_from_declared_equations",
            },
            "continuation": {"used": True, "eta_path": eta_path, "continuation_mode": "integer", "memory_window_propagated": None, "final_eta": 1.0},
            "trajectory": {"bounded": True, "nontrivial": True, "finite_fraction": 1.0, "post_transient_rows": post_count, "minimum_post_transient_length": 1000},
            "robustness": robustness,
            "hiddenness": {
                "tested_all_equilibria": True,
                "tested_radii": [1.0e-5, 1.0e-3, 1.0e-2],
                "required_radii": [1.0e-5, 1.0e-3, 1.0e-2],
                "target_hits_from_equilibria": 0,
                "basin_intersection_detected": False,
                "basin_controls_complete": True,
                "coverage_by_equilibrium_radius_complete": True,
                "numerical_failures": 0,
            },
            "lyapunov": {"exponents": strict_exponents, "method_status": "internal_controls_passed"},
            "zero_one": {"K": 0.985, "state": "zero_one_chaotic_candidate", "gate_applicable": True, "series": "Poincare return sequence"},
            "spectrum": {"state_global": "spectral_inconclusive", "gate_applicable": False},
            "poincare": {
                **{
                    key: value
                    for key, value in poincare_summary.items()
                    if key != "section_metadata"
                },
                "gate_applicable": True,
            },
            "tolerances": gate_tolerances,
        },
    }
    _write_json(run_dir / "09_candidate_gate.json", gate_file)

    figures_dir = run_dir / "figures"
    figures_dir.mkdir()
    global_root = figure_store_root or (repo / "outputs" / "library_figures")
    local_figure_rows = []
    global_figure_rows = []
    for figure_id in module.FIGURE_IDS:
        parameters = dict(module.EXPECTED_BASE_PARAMETERS) if figure_id.startswith("00_") else candidate_parameters
        metadata = {
            "caption_key": f"fig_{figure_id}",
            "source_script": module.SCIENTIFIC_SOURCE_FIXED_FILES[0],
            "source_function": "run_contract/run_diagnostics/run_hiddenness",
            "data_sources": module.FIGURE_DATA_SOURCES[figure_id],
            "system_id": "modified_van_der_pol_duffing",
            "q": 1.0,
            "parameters": parameters,
            "integrator": "display_only" if figure_id.startswith("00_") else "DOP853",
            "memory_mode": "not_applicable_integer_q1",
            "t_final": 900.0,
            "t_burn": 300.0,
            "scientific_source_bundle_sha256": source["bundle_sha256"],
            "quick_smoke_only": False,
        }
        local_png = figures_dir / f"{figure_id}.png"
        local_pdf = figures_dir / f"{figure_id}.pdf"
        local_png.write_bytes(b"\x89PNG\r\n\x1a\nfixture-IEND\xaeB`\x82")
        local_pdf.write_bytes(b"%PDF-1.4\nfixture\n%%EOF\n")
        central_paths = {}
        for suffix, local_path in (("png", local_png), ("pdf", local_pdf)):
            for target in (
                global_root / "by_run" / run_id / suffix / f"{figure_id}.{suffix}",
                global_root / "current" / suffix / f"{figure_id}.{suffix}",
                global_root / "by_export" / "mavpd_integer_hidden_chaos_report" / suffix / f"{figure_id}.{suffix}",
            ):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(local_path.read_bytes())
            central_paths[suffix] = str(global_root / "by_run" / run_id / suffix / f"{figure_id}.{suffix}")
        metadata_path = global_root / "by_run" / run_id / "metadata" / f"{figure_id}.json"
        _write_json(metadata_path, metadata)
        local_figure_rows.append(
            {
                "figure_id": figure_id,
                "run_id": run_id,
                "local_png": f"figures/{figure_id}.png",
                "local_pdf": f"figures/{figure_id}.pdf",
                "png_sha256": _digest(local_png),
                "pdf_sha256": _digest(local_pdf),
                "global_promotion_requested": True,
                "promoted_to_global_manifest": True,
                "central_paths": central_paths,
                "metadata": metadata,
            }
        )
        global_figure_rows.append(
            {
                "figure_id": figure_id,
                "caption_key": metadata["caption_key"],
                "kind": "mavpd_integer_hidden_chaos",
                "source_script": metadata["source_script"],
                "source_function": metadata["source_function"],
                "data_sources": metadata["data_sources"],
                "run_id": run_id,
                "system_id": metadata["system_id"],
                "q": metadata["q"],
                "parameters": metadata["parameters"],
                "integrator": metadata["integrator"],
                "memory_mode": metadata["memory_mode"],
                "t_final": metadata["t_final"],
                "t_burn": metadata["t_burn"],
                "pdf_path": f"library_figures/by_run/{run_id}/pdf/{figure_id}.pdf",
                "png_path": f"library_figures/by_run/{run_id}/png/{figure_id}.png",
                "metadata_path": f"library_figures/by_run/{run_id}/metadata/{figure_id}.json",
                "created_at": "2026-08-03T12:00:00+00:00",
                "git_commit": "test",
                "export_targets": ["mavpd_integer_hidden_chaos_report"],
            }
        )
    global_manifest_json = global_root / "manifests" / "figure_manifest.json"
    global_manifest_csv = global_root / "manifests" / "figure_manifest.csv"
    _write_json(global_manifest_json, global_figure_rows)
    _write_csv(
        global_manifest_csv,
        [
            {
                field: json.dumps(row.get(field)) if isinstance(row.get(field), (list, dict)) else str(row.get(field, ""))
                for field in module.GLOBAL_MANIFEST_FIELDS
            }
            for row in global_figure_rows
        ],
        list(module.GLOBAL_MANIFEST_FIELDS),
    )
    receipt = {
        "status": "committed",
        "run_id": run_id,
        "scientific_source_bundle_sha256": source["bundle_sha256"],
        "figure_count": 8,
        "figure_ids": list(module.FIGURE_IDS),
        "seconds": 0.25,
        "global_manifest_paths": [str(global_manifest_json), str(global_manifest_csv)],
    }
    timings = [
        {"phase": "contract", "seconds": 1.0, "timing_source": "perf_counter"},
        {"phase": "search", "seconds": 2.0, "timing_source": "perf_counter"},
        {"phase": "diagnostics", "seconds": 2.0, "timing_source": "perf_counter"},
        {"phase": "hiddenness", "seconds": 2.0, "timing_source": "perf_counter"},
        {"phase": "candidate_gate", "seconds": 1.25, "timing_source": "perf_counter"},
        {"phase": "total", "seconds": 10.0, "timing_source": "perf_counter"},
    ]
    _write_json(figures_dir / "figure_manifest.json", {"figures": local_figure_rows})
    _write_json(figures_dir / "global_promotion_receipt.json", receipt)
    _write_csv(run_dir / "phase_timings.csv", timings)
    manifest = {
        "case_id": module.CASE_ID,
        "run_id": run_id,
        "config_sha256": config_sha,
        "runtime_environment": runtime,
        "quick_mode": False,
        "scientific_source_snapshot": source,
        "candidate_parameters": candidate_parameters,
        "hopf_boundary": hopf,
        "selected_hopf_offset": selected_offset,
        "alternative_source_branch_index": 0,
        "alternative_source_branch_selection_rule": "lowest direct harmonic frequency among successful direct base branches",
        "candidate_parameter_provenance": "gamma_selected_at_declared_xi_continuation_endpoint_not_a_published_parameter_tuple",
        "xi_endpoint_provenance": "declared_local_continuation_endpoint_not_selected_by_candidate_screen",
        "lyapunov_exponents": strict_exponents,
        "candidate_gate": gate,
        "hiddenness": hidden_summary,
        "figures": local_figure_rows,
        "frequency_grid_used_for_search": False,
        "global_proof_claimed": False,
        "global_figure_promotion": receipt,
        "timings": timings,
    }
    _write_json(run_dir / "run_manifest.json", manifest)

    # Every semantic artifact above is bound by the final run ledger.
    ledger = {relative: _digest(run_dir / relative) for relative in sorted(module.REQUIRED_LEDGER_ARTIFACTS)}
    status = {
        "status": "complete",
        "run_id": run_id,
        "created_at_utc": "2026-08-03T12:00:00+00:00",
        "completed_at_utc": "2026-08-03T12:10:00+00:00",
        "quick_mode": False,
        "config_sha256": config_sha,
        "runtime_environment": runtime,
        "scientific_source_snapshot": source,
        "completed_phases": list(module.DIRECT_PHASES),
        "last_completed_phase": "global_figure_promotion",
        "artifacts": ledger,
    }
    _write_json(run_dir / "run_status.json", status)
    return module, repo, run_dir


@pytest.mark.unit
def test_valid_full_run_passes_without_writing(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    before = _tree_state(repo)

    summary = module.validate_run(run_dir, repo, require_active_promotion=True)

    assert summary["ok"] is True, summary["errors"]
    assert summary["total_probes"] == 112
    assert summary["probe_counts"] == {"main_3x3xN": 108, "targeted_E0_unstable": 4}
    assert summary["local_figure_pairs"] == 8
    assert summary["local_figure_hashes_validated"] == 16
    assert summary["global_manifest_entries"] == 8
    assert summary["frequency_sweep_used"] is False
    assert _tree_state(repo) == before


@pytest.mark.unit
def test_valid_resumed_run_may_retain_superseded_direct_hiddenness_phase(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    timings = [
        {"phase": "contract", "seconds": 1.0, "timing_source": "perf_counter"},
        {"phase": "search", "seconds": 2.0, "timing_source": "perf_counter"},
        {"phase": "diagnostics", "seconds": 2.0, "timing_source": "perf_counter"},
        {"phase": "hiddenness_resumed", "seconds": 2.0, "timing_source": "perf_counter"},
        {"phase": "candidate_gate_resumed", "seconds": 1.25, "timing_source": "perf_counter"},
        {
            "phase": "total_recorded_phase_seconds",
            "seconds": 8.25,
            "timing_source": "sum_of_persisted_and_resumed_phase_timers_not_wall_clock",
        },
    ]
    _write_csv(run_dir / "phase_timings.csv", timings)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("quick_mode")
    manifest["resumed_from_persisted_phase_record"] = True
    manifest["timings"] = timings
    _write_json(manifest_path, manifest)
    status_path = run_dir / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["completed_phases"] = list(module.DIRECT_PHASES[:4] + module.RESUMED_PHASE_TAIL)
    _write_json(status_path, status)
    _rebind(run_dir, "phase_timings.csv", "run_manifest.json")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is True, summary["errors"]


@pytest.mark.unit
def test_valid_route_selects_branch_one_when_lower_frequency_branch_fails(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    route_path = run_dir / "01_direct_seed_and_lambda_continuation.json"
    route = json.loads(route_path.read_text(encoding="utf-8"))
    failed = route["branches"][0]["lyapunov"]
    failed["status"] = "diverged"
    failed["t_accumulate"] = 100.0
    failed["metadata"]["t_accumulate_completed"] = 100.0
    failed["metadata"]["qr_segments"] = 200
    route["alternative_source_branch_index"] = 1
    _write_json(route_path, route)

    selection_path = run_dir / "03_candidate_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["alternative_source_branch_index"] = 1
    _write_json(selection_path, selection)

    continuation_path = run_dir / "02_parameter_continuation.csv"
    with continuation_path.open("r", newline="", encoding="utf-8") as handle:
        continuation = list(csv.DictReader(handle))
    first_xi = next(row for row in continuation if row["stage"] == "xi")
    first_xi["x_in"] = json.dumps(route["branches"][1]["final_state"])
    _write_csv(continuation_path, continuation)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["alternative_source_branch_index"] = 1
    _write_json(manifest_path, manifest)

    contract = json.loads((run_dir / "00_system_contract.json").read_text(encoding="utf-8"))
    branch_one_seed = contract["direct_seed_records"][1]
    gate_path = run_dir / "09_candidate_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    seed_parameters = gate["evidence"]["run_metadata"]["seed"]["parameters"]
    seed_parameters.update(
        {
            "source_branch_index": 1,
            "theoretical_harmonic_seed": branch_one_seed["seed"],
            "omega0": branch_one_seed["omega0"],
            "k": branch_one_seed["k"],
            "a0": branch_one_seed["a0"],
        }
    )
    _write_json(gate_path, gate)
    _rebind(
        run_dir,
        "01_direct_seed_and_lambda_continuation.json",
        "02_parameter_continuation.csv",
        "03_candidate_selection.json",
        "09_candidate_gate.json",
        "run_manifest.json",
    )

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is True, summary["errors"]


@pytest.mark.unit
def test_valid_zero_one_gate_uses_median_threshold_not_each_result_state(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    return_path = run_dir / "05_zero_one_return_map.json"
    return_map = json.loads(return_path.read_text(encoding="utf-8"))
    for row, k_value in zip(return_map, (0.75, 0.65)):
        row.update(
            {
                "K": k_value,
                "K_median": k_value,
                "K_mean": k_value,
                "K_min": k_value - 0.01,
                "K_max": k_value + 0.005,
                "state": "zero_one_inconclusive",
            }
        )
    _write_json(return_path, return_map)
    diagnostics_path = run_dir / "05_chaos_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["zero_one"]["return_map_results"] = return_map
    _write_json(diagnostics_path, diagnostics)
    gate_path = run_dir / "09_candidate_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["evidence"]["zero_one"]["K"] = 0.7
    gate["evidence"]["zero_one"]["state"] = "zero_one_chaotic_candidate"
    _write_json(gate_path, gate)
    _rebind(
        run_dir,
        "05_zero_one_return_map.json",
        "05_chaos_diagnostics.json",
        "09_candidate_gate.json",
    )

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is True, summary["errors"]


@pytest.mark.unit
def test_valid_joint_gate_accepts_poincare_support_when_zero_one_is_intermediate(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    return_path = run_dir / "05_zero_one_return_map.json"
    return_map = json.loads(return_path.read_text(encoding="utf-8"))
    for row in return_map:
        row.update(
            {
                "K": 0.5,
                "K_median": 0.5,
                "K_mean": 0.5,
                "K_min": 0.49,
                "K_max": 0.505,
                "state": "zero_one_inconclusive",
            }
        )
    _write_json(return_path, return_map)
    diagnostics_path = run_dir / "05_chaos_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["zero_one"]["return_map_results"] = return_map
    _write_json(diagnostics_path, diagnostics)
    gate_path = run_dir / "09_candidate_gate.json"
    gate_file = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_file["evidence"]["zero_one"]["K"] = 0.5
    gate_file["evidence"]["zero_one"]["state"] = "zero_one_inconclusive"
    gate_file["gate"]["zero_one_support"] = "not_available_or_intermediate"
    _write_json(gate_path, gate_file)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate_gate"] = gate_file["gate"]
    _write_json(manifest_path, manifest)
    _rebind(
        run_dir,
        "05_zero_one_return_map.json",
        "05_chaos_diagnostics.json",
        "09_candidate_gate.json",
        "run_manifest.json",
    )

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is True, summary["errors"]


@pytest.mark.unit
def test_detects_ledger_tampering_and_current_source_drift(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    (run_dir / "05_chaos_diagnostics.json").write_text('{"tampered":true}\n', encoding="utf-8")
    (repo / "hidden_attractors" / "solver.py").write_text("METHOD = 'changed'\n", encoding="utf-8")

    summary = module.validate_run(run_dir, repo, require_active_promotion=True)

    assert summary["ok"] is False
    assert summary["checks"]["ledger"]["ok"] is False
    assert summary["checks"]["scientific_sources"]["ok"] is False


@pytest.mark.unit
def test_rejects_failed_joint_gate_frequency_sweep_and_global_claim(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    contract = json.loads((run_dir / "00_system_contract.json").read_text(encoding="utf-8"))
    contract["frequency_grid_used_for_seed"] = True
    _write_json(run_dir / "00_system_contract.json", contract)
    gate_file = json.loads((run_dir / "09_candidate_gate.json").read_text(encoding="utf-8"))
    gate_file["gate"]["chaotic_hidden_promotion_allowed"] = False
    _write_json(run_dir / "09_candidate_gate.json", gate_file)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["candidate_gate"] = gate_file["gate"]
    manifest["global_proof_claimed"] = True
    _write_json(run_dir / "run_manifest.json", manifest)
    _rebind(
        run_dir,
        "00_system_contract.json",
        "09_candidate_gate.json",
        "run_manifest.json",
    )

    summary = module.validate_run(run_dir, repo, require_active_promotion=True)

    assert summary["ok"] is False
    assert summary["checks"]["joint_gate"]["ok"] is False
    assert summary["checks"]["no_frequency_sweep"]["ok"] is False
    assert summary["checks"]["finite_scope"]["ok"] is False


@pytest.mark.unit
def test_rejects_incomplete_108_plus_4_csv_even_when_rebound(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    path = run_dir / "07_hiddenness_probes.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _write_csv(path, rows[:-1])
    _rebind(run_dir, "07_hiddenness_probes.csv")

    summary = module.validate_run(run_dir, repo, require_active_promotion=True)

    assert summary["ok"] is False
    assert summary["checks"]["ledger"]["ok"] is True
    assert summary["checks"]["probe_coverage"]["ok"] is False


@pytest.mark.unit
def test_rejects_global_copy_mismatch_and_timing_without_promotion(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    run_id = status["run_id"]
    figure_id = module.FIGURE_IDS[0]
    current = repo / "outputs" / "library_figures" / "current" / "png" / f"{figure_id}.png"
    current.write_bytes(b"wrong-current-copy")

    receipt_path = run_dir / "figures" / "global_promotion_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["seconds"] = 2.0
    _write_json(receipt_path, receipt)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["global_figure_promotion"] = receipt
    _write_json(manifest_path, manifest)
    _rebind(run_dir, "figures/global_promotion_receipt.json", "run_manifest.json")

    summary = module.validate_run(run_dir, repo, require_active_promotion=True)

    assert summary["ok"] is False
    assert summary["checks"]["global_figures"]["ok"] is False
    assert summary["checks"]["timings"]["ok"] is False


@pytest.mark.unit
def test_cli_emits_json_and_returns_nonzero_on_failure(tmp_path: Path, capsys) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    status_path = run_dir / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["quick_mode"] = True
    _write_json(status_path, status)

    exit_code = module.main(["--run-dir", str(run_dir), "--repo-root", str(repo)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["read_only"] is True
    assert payload["checks"]["status"]["ok"] is False


@pytest.mark.unit
def test_cli_exception_handler_does_not_reresolve_a_hostile_path(monkeypatch, capsys) -> None:
    module = _module()

    def hostile_resolve(_path):
        raise RuntimeError("synthetic resolve failure")

    monkeypatch.setattr(module.Path, "resolve", hostile_resolve)

    exit_code = module.main(["--run-dir", "hostile-run-path"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["run_dir"] == "hostile-run-path"
    assert payload["checks"]["internal_validation"]["ok"] is False
    assert "RuntimeError: synthetic resolve failure" in payload["errors"][0]["message"]


@pytest.mark.unit
def test_rejects_hash_rebound_placeholder_dynamics_and_forged_strong_gate(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    _write_json(run_dir / "05_chaos_diagnostics.json", {})
    _rebind(run_dir, "05_chaos_diagnostics.json")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["finite_dynamics"]["ok"] is False
    assert summary["checks"]["joint_gate_semantics"]["ok"] is False


@pytest.mark.unit
@pytest.mark.parametrize("invalid_number", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_rejects_nonfinite_json_even_when_ledger_is_rebound(
    tmp_path: Path,
    invalid_number: str,
) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    path = run_dir / "05_chaos_diagnostics.json"
    path.write_text(
        f'{{"candidate_parameters": {{"gamma": {invalid_number}}}}}\n',
        encoding="utf-8",
    )
    _rebind(run_dir, "05_chaos_diagnostics.json")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["json_parse"]["ok"] is False


@pytest.mark.unit
def test_rejects_duplicate_json_keys_even_when_ledger_is_rebound(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    path = run_dir / "05_chaos_diagnostics.json"
    path.write_text(
        '{"candidate_parameters": {}, "candidate_parameters": {}}\n',
        encoding="utf-8",
    )
    _rebind(run_dir, "05_chaos_diagnostics.json")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["json_parse"]["ok"] is False


@pytest.mark.unit
def test_rejects_boolean_used_as_numeric_evidence(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    path = run_dir / "05_chaos_diagnostics.json"
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    diagnostics["boundedness"]["post_transient_rows"] = True
    _write_json(path, diagnostics)
    _rebind(run_dir, "05_chaos_diagnostics.json")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["finite_dynamics"]["ok"] is False


@pytest.mark.unit
def test_rejects_boolean_alias_for_source_branch_across_lineage(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    route_path = run_dir / "01_direct_seed_and_lambda_continuation.json"
    route = json.loads(route_path.read_text(encoding="utf-8"))
    route["alternative_source_branch_index"] = False
    _write_json(route_path, route)
    selection_path = run_dir / "03_candidate_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["alternative_source_branch_index"] = False
    _write_json(selection_path, selection)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["alternative_source_branch_index"] = False
    _write_json(manifest_path, manifest)
    _rebind(
        run_dir,
        "01_direct_seed_and_lambda_continuation.json",
        "03_candidate_selection.json",
        "run_manifest.json",
    )

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["candidate_lineage"]["ok"] is False


@pytest.mark.unit
def test_rejects_placeholder_screening_contract_even_when_rebound(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    path = run_dir / "03_candidate_screening_contract.json"
    _write_json(path, {})
    _rebind(run_dir, "03_candidate_screening_contract.json")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["candidate_lineage"]["ok"] is False


@pytest.mark.unit
def test_rejects_screening_non_target_label_inside_calibrated_ambiguity_band(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    path = run_dir / "03_candidate_screening_probes.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["target_distance_norm"] = "0.011"
    _write_csv(path, rows)
    _rebind(run_dir, "03_candidate_screening_probes.csv")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["candidate_lineage"]["ok"] is False


@pytest.mark.unit
def test_rejects_incomplete_gamma_continuation_even_when_rebound(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    path = run_dir / "02_parameter_continuation.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    gamma_indices = [index for index, row in enumerate(rows) if row["stage"] == "gamma"]
    assert gamma_indices
    del rows[gamma_indices[len(gamma_indices) // 2]]
    _write_csv(path, rows)
    _rebind(run_dir, "02_parameter_continuation.csv")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["candidate_lineage"]["ok"] is False


@pytest.mark.unit
def test_rejects_short_csv_row_even_when_ledger_is_rebound(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    path = run_dir / "04_candidate_trajectory.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    lines[1] = ",".join(fields[:-1])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rebind(run_dir, "04_candidate_trajectory.csv")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["csv_parse"]["ok"] is False


@pytest.mark.unit
def test_rejects_selection_state_not_from_continuation(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    selection_path = run_dir / "03_candidate_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["selected_candidate_initial_state"] = [9.0, 8.0, 7.0]
    _write_json(selection_path, selection)
    _rebind(run_dir, "03_candidate_selection.json")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["candidate_lineage"]["ok"] is False


@pytest.mark.unit
def test_rejects_arbitrary_points_labeled_as_equilibrium_neighborhoods(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    probe_path = run_dir / "07_hiddenness_probes.csv"
    initial_path = run_dir / "07_hiddenness_initial_conditions.csv"
    with probe_path.open("r", newline="", encoding="utf-8") as handle:
        probes = list(csv.DictReader(handle))
    with initial_path.open("r", newline="", encoding="utf-8") as handle:
        initial = list(csv.DictReader(handle))
    probes[0]["x0"] = json.dumps([10.0, 10.0, 10.0])
    initial[0]["y1"], initial[0]["y2"], initial[0]["y3"] = 10.0, 10.0, 10.0
    _write_csv(probe_path, probes)
    _write_csv(initial_path, initial)
    _rebind(run_dir, "07_hiddenness_probes.csv", "07_hiddenness_initial_conditions.csv")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["probe_geometry"]["ok"] is False


@pytest.mark.unit
def test_malformed_embedded_probe_vector_is_reported_without_crashing(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    probe_path = run_dir / "07_hiddenness_probes.csv"
    with probe_path.open("r", newline="", encoding="utf-8") as handle:
        probes = list(csv.DictReader(handle))
    probes[0]["x0"] = json.dumps([[], [], []])
    _write_csv(probe_path, probes)
    _rebind(run_dir, "07_hiddenness_probes.csv")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["probe_initial_conditions"]["ok"] is False


@pytest.mark.unit
def test_rejects_boolean_alias_for_return_map_coordinate(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    return_path = run_dir / "05_zero_one_return_map.json"
    return_map = json.loads(return_path.read_text(encoding="utf-8"))
    return_map[0]["coordinate"] = False
    _write_json(return_path, return_map)
    diagnostics_path = run_dir / "05_chaos_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["zero_one"]["return_map_results"] = return_map
    _write_json(diagnostics_path, diagnostics)
    _rebind(run_dir, "05_zero_one_return_map.json", "05_chaos_diagnostics.json")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["finite_dynamics"]["ok"] is False


@pytest.mark.unit
def test_rejects_invalid_image_signatures_after_all_hashes_are_rebound(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    run_id = status["run_id"]
    figure_id = module.FIGURE_IDS[0]
    local_path = run_dir / "figures" / f"{figure_id}.png"
    invalid = b"not-a-png-but-hash-consistent"
    local_path.write_bytes(invalid)
    global_root = repo / "outputs" / "library_figures"
    for path in (
        global_root / "by_run" / run_id / "png" / f"{figure_id}.png",
        global_root / "current" / "png" / f"{figure_id}.png",
        global_root / "by_export" / "mavpd_integer_hidden_chaos_report" / "png" / f"{figure_id}.png",
    ):
        path.write_bytes(invalid)
    local_manifest_path = run_dir / "figures" / "figure_manifest.json"
    local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
    local_manifest["figures"][0]["png_sha256"] = _digest(local_path)
    _write_json(local_manifest_path, local_manifest)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["figures"] = local_manifest["figures"]
    _write_json(run_manifest_path, run_manifest)
    _rebind(
        run_dir,
        f"figures/{figure_id}.png",
        "figures/figure_manifest.json",
        "run_manifest.json",
    )

    summary = module.validate_run(run_dir, repo, require_active_promotion=True)

    assert summary["ok"] is False
    assert summary["checks"]["figures"]["ok"] is False
    assert any("valid-signature PNG" in error["message"] for error in summary["errors"])


@pytest.mark.unit
def test_rejects_invalid_pdf_signature_after_all_hashes_are_rebound(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    run_id = status["run_id"]
    figure_id = module.FIGURE_IDS[0]
    local_path = run_dir / "figures" / f"{figure_id}.pdf"
    invalid = b"not-a-pdf-but-hash-consistent"
    local_path.write_bytes(invalid)
    global_root = repo / "outputs" / "library_figures"
    for path in (
        global_root / "by_run" / run_id / "pdf" / f"{figure_id}.pdf",
        global_root / "current" / "pdf" / f"{figure_id}.pdf",
        global_root / "by_export" / "mavpd_integer_hidden_chaos_report" / "pdf" / f"{figure_id}.pdf",
    ):
        path.write_bytes(invalid)
    local_manifest_path = run_dir / "figures" / "figure_manifest.json"
    local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
    local_manifest["figures"][0]["pdf_sha256"] = _digest(local_path)
    _write_json(local_manifest_path, local_manifest)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["figures"] = local_manifest["figures"]
    _write_json(run_manifest_path, run_manifest)
    _rebind(
        run_dir,
        f"figures/{figure_id}.pdf",
        "figures/figure_manifest.json",
        "run_manifest.json",
    )

    summary = module.validate_run(run_dir, repo, require_active_promotion=True)

    assert summary["ok"] is False
    assert summary["checks"]["figures"]["ok"] is False
    assert any("valid-signature PDF" in error["message"] for error in summary["errors"])


@pytest.mark.unit
def test_rejects_non_target_hiddenness_label_inside_ambiguity_band(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    path = run_dir / "07_hiddenness_probes.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["target_distance_norm"] = "0.009"
    _write_csv(path, rows)
    _rebind(run_dir, "07_hiddenness_probes.csv")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["probe_outcomes"]["ok"] is False


@pytest.mark.unit
def test_active_manifest_csv_divergence_is_optional_but_detected_when_required(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    csv_path = repo / "outputs" / "library_figures" / "manifests" / "figure_manifest.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["kind"] = "corrupt-kind"
    _write_csv(csv_path, rows, list(module.GLOBAL_MANIFEST_FIELDS))

    immutable = module.validate_run(run_dir, repo)
    active = module.validate_run(run_dir, repo, require_active_promotion=True)

    assert immutable["ok"] is True, immutable["errors"]
    assert immutable["active_promotion_checked"] is False
    assert active["ok"] is False
    assert active["checks"]["global_manifest"]["ok"] is False


@pytest.mark.unit
def test_rejects_noncanonical_local_figure_path_even_if_file_exists(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    local_manifest_path = run_dir / "figures" / "figure_manifest.json"
    local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
    local_manifest["figures"][0]["local_png"] = local_manifest["figures"][0]["local_png"].replace("/", "\\")
    _write_json(local_manifest_path, local_manifest)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["figures"] = local_manifest["figures"]
    _write_json(run_manifest_path, run_manifest)
    _rebind(run_dir, "figures/figure_manifest.json", "run_manifest.json")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["figures"]["ok"] is False


@pytest.mark.unit
def test_rejects_runtime_and_direct_resume_mode_mismatch(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_environment"]["numpy_version"] = "different"
    manifest["resumed_from_persisted_phase_record"] = True
    _write_json(manifest_path, manifest)
    _rebind(run_dir, "run_manifest.json")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["runtime"]["ok"] is False
    assert summary["checks"]["status"]["ok"] is False


@pytest.mark.unit
def test_rejects_run_id_whose_source_prefix_does_not_match_snapshot(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    status_path = run_dir / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    actual_prefix = status["scientific_source_snapshot"]["bundle_sha256"][:12]
    wrong_prefix = ("0" if actual_prefix[0] != "0" else "1") + actual_prefix[1:]
    status["run_id"] = (
        "mavpd_integer_hidden_chaos-full-20260803T120000000000Z-"
        f"{wrong_prefix}-0123456789ab"
    )
    _write_json(status_path, status)

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["status"]["ok"] is False
    assert any("source prefix" in error["message"] for error in summary["errors"])


@pytest.mark.unit
def test_separate_scientific_source_root_and_snapshot_manifest(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    source_root = tmp_path / "immutable-source"
    for relative, expected_hash in status["scientific_source_snapshot"]["files"].items():
        source_path = repo / relative
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_path.read_bytes())
        assert _digest(destination) == expected_hash
    snapshot_manifest = {
        "algorithm": "sha256",
        "bundle_sha256": status["scientific_source_snapshot"]["bundle_sha256"],
        "created_at_utc": "2026-08-03T12:00:00Z",
        "file_count": len(status["scientific_source_snapshot"]["files"]),
        "files": {
            relative: {
                "sha256": digest,
                "size_bytes": (source_root / relative).stat().st_size,
            }
            for relative, digest in status["scientific_source_snapshot"]["files"].items()
        },
    }
    _write_json(source_root / "snapshot_manifest.json", snapshot_manifest)
    (repo / "hidden_attractors" / "solver.py").write_text("METHOD = 'changed-after-snapshot'\n", encoding="utf-8")

    default_root = module.validate_run(run_dir, repo)
    immutable_root = module.validate_run(run_dir, repo, source_root)

    assert default_root["ok"] is False
    assert default_root["checks"]["scientific_sources"]["ok"] is False
    assert immutable_root["ok"] is True, immutable_root["errors"]
    assert immutable_root["snapshot_manifests_validated"] == [str((source_root / "snapshot_manifest.json").resolve())]


@pytest.mark.unit
def test_separate_configured_figure_store_root_is_resolved_and_confined(tmp_path: Path) -> None:
    external_store = tmp_path / "configured_outputs" / "library_figures"
    module, repo, run_dir = _build_valid_run(tmp_path, figure_store_root=external_store)

    default_store = module.validate_run(run_dir, repo)
    configured_store = module.validate_run(
        run_dir,
        repo,
        figure_store_root=external_store,
        require_active_promotion=True,
    )

    assert default_store["ok"] is False
    assert default_store["checks"]["paths"]["ok"] is False
    assert configured_store["ok"] is True, configured_store["errors"]
    assert configured_store["figure_store_root"] == str(external_store.resolve())


@pytest.mark.unit
def test_rejects_snapshot_manifest_that_disagrees_with_status(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    bad_snapshot = dict(status["scientific_source_snapshot"])
    bad_snapshot["bundle_sha256"] = "0" * 64
    _write_json(run_dir / "snapshot_manifest.json", bad_snapshot)

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["snapshot_manifest"]["ok"] is False


@pytest.mark.unit
def test_rejects_regular_zero_one_median_rebound_as_strong_chaos(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    return_path = run_dir / "05_zero_one_return_map.json"
    return_map = json.loads(return_path.read_text(encoding="utf-8"))
    for row in return_map:
        row.update(
            {
                "K": 0.25,
                "K_median": 0.25,
                "K_mean": 0.25,
                "K_min": 0.24,
                "K_max": 0.255,
                "K_std": 0.005,
                "state": "zero_one_inconclusive",
            }
        )
    _write_json(return_path, return_map)
    diagnostics_path = run_dir / "05_chaos_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["zero_one"]["return_map_results"] = return_map
    _write_json(diagnostics_path, diagnostics)
    gate_path = run_dir / "09_candidate_gate.json"
    gate_file = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_file["evidence"]["zero_one"].update({"K": 0.25, "state": "zero_one_inconclusive"})
    gate_file["gate"]["zero_one_support"] = "not_available_or_intermediate"
    _write_json(gate_path, gate_file)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate_gate"] = gate_file["gate"]
    _write_json(manifest_path, manifest)
    _rebind(
        run_dir,
        "05_zero_one_return_map.json",
        "05_chaos_diagnostics.json",
        "09_candidate_gate.json",
        "run_manifest.json",
    )

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["joint_gate_semantics"]["ok"] is False
    assert any("regularity conflict" in error["message"] for error in summary["errors"])


@pytest.mark.unit
def test_rejects_rebound_gate_tolerance_contract_drift(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    gate_path = run_dir / "09_candidate_gate.json"
    gate_file = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_file["evidence"]["run_metadata"]["tolerances"]["zero_one_regular_threshold"] = 0.25
    gate_file["evidence"]["tolerances"]["zero_one_regular_threshold"] = 0.25
    _write_json(gate_path, gate_file)
    _rebind(run_dir, "09_candidate_gate.json")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["joint_gate_semantics"]["ok"] is False
    assert any("frozen full-run contract" in error["message"] for error in summary["errors"])


@pytest.mark.unit
def test_rejects_boolean_aliases_in_rebound_robustness_contract(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    robustness_path = run_dir / "08_robustness_matrix.json"
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    robustness["tested_h"] = 1
    _write_json(robustness_path, robustness)
    gate_path = run_dir / "09_candidate_gate.json"
    gate_file = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_file["evidence"]["robustness"]["tested_h"] = 1
    _write_json(gate_path, gate_file)
    _rebind(run_dir, "08_robustness_matrix.json", "09_candidate_gate.json")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["finite_dynamics"]["ok"] is False
    assert summary["checks"]["joint_gate_semantics"]["ok"] is False


@pytest.mark.unit
def test_rejects_negative_efork_distances_even_when_median_is_self_consistent(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    diagnostics_path = run_dir / "05_chaos_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    target_match = diagnostics["efork_crosscheck"]["target_match"]
    target_match["distances_norm"] = [-0.003, -0.002, -0.001]
    target_match["distance_norm"] = -0.002
    _write_json(diagnostics_path, diagnostics)
    _rebind(run_dir, "05_chaos_diagnostics.json")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["finite_dynamics"]["ok"] is False
    assert any("EFORK target distance" in error["message"] for error in summary["errors"])


@pytest.mark.unit
def test_rejects_rebound_lyapunov_solver_metadata_drift(tmp_path: Path) -> None:
    module, repo, run_dir = _build_valid_run(tmp_path)
    diagnostics_path = run_dir / "05_chaos_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["lyapunov"]["metadata"]["jacobian_source"] = "finite_difference"
    _write_json(diagnostics_path, diagnostics)
    _rebind(run_dir, "05_chaos_diagnostics.json")

    summary = module.validate_run(run_dir, repo)

    assert summary["ok"] is False
    assert summary["checks"]["finite_dynamics"]["ok"] is False


@pytest.mark.unit
def test_characteristic_residual_handles_extreme_finite_input_without_overflow() -> None:
    module = _module()

    residual = module._characteristic_residual(complex(1.0e308, 0.0), (1.0, 2.0, 3.0))

    assert math.isfinite(residual)
    assert residual > 0.0
