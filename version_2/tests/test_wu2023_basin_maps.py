"""Method dispatch, terminal interpretation and provenance of Wu map slices."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hidden_attractors.integrations.adm_wu2023 import adm_wu2023_integrate
from hidden_attractors.validation import wu2023_basin_maps as maps


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "validation/reference_cases/fractional_chua_arctan_wu2023/config.json"


def tiny_contract():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["numerical_contract"].update(N=10, t_final=.1)
    config["basin_map"] = {"points": 3, "limit": .1, "planes": ["xy"], "t_burn": .05, "terminal_window_points": 3}
    return config


def write_config(tmp_path, config):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_adm_dispatch_preserves_declared_parameters_and_fractional_order(monkeypatch):
    config = tiny_contract()
    config["parameters"]["alpha"] = 9.21
    config["numerical_contract"]["q"] = np.nextafter(1., 0.)
    validated = maps.validate_map_contract(config)
    seen = {}
    def specialized(params, initial, **options):
        seen.update(params=copy.deepcopy(params), initial=np.array(initial), options=options)
        return np.array([0., .1]), np.zeros((2, 3)), "ok", {}
    monkeypatch.setattr(maps, "adm_wu2023_integrate", specialized)
    from hidden_attractors.integrations import selector
    monkeypatch.setattr(selector, "integrate", lambda *a, **k: pytest.fail("ADM silently dispatched to a Caputo facade"))
    maps.integrate_map_point(validated, np.array([.2, -.3, .4]))
    assert seen["params"]["alpha"] == 9.21
    assert seen["params"]["a2"] == -1.5585
    assert seen["options"] == {"q": np.nextafter(1., 0.), "h": .01, "N": 10, "divergence_norm": 120.}
    np.testing.assert_array_equal(seen["initial"], [.2, -.3, .4])


@pytest.mark.parametrize(("section", "key", "value"), [
    ("numerical_contract", "adm_order", 3),
    ("numerical_contract", "memory_length", 100.),
    ("numerical_contract", "memory_policy", "full"),
    ("numerical_contract", "caputo_history_accumulated", True),
    ("numerical_contract", "backend", "efork3"),
    ("numerical_contract", "N", 11),
    ("numerical_contract", "q", float("nan")),
    ("parameters", "n", -1.),
    ("parameters", "rho", 0.),
    ("basin_map", "planes", ["xy", "xy"]),
    ("basin_map", "terminal_window_points", 9),
    ("basin_map", "points", True),
])
def test_invalid_contract_creates_no_output(tmp_path, section, key, value):
    config = tiny_contract()
    config[section][key] = value
    source = write_config(tmp_path, config)
    output = tmp_path / "must_not_exist"
    with pytest.raises((ValueError, TypeError)):
        maps.run_basin_maps(source, output)
    assert not output.exists()


def test_output_guard_precedes_every_integration_and_preserves_files(tmp_path, monkeypatch):
    source = write_config(tmp_path, tiny_contract())
    output = tmp_path / "old_run"
    output.mkdir()
    evidence = output / "old.csv"
    evidence.write_bytes(b"irreplaceable\r\n")
    monkeypatch.setattr(maps, "integrate_map_point", lambda *a: pytest.fail("integration started before output guard"))
    with pytest.raises(FileExistsError, match="not empty"):
        maps.run_basin_maps(source, output)
    assert evidence.read_bytes() == b"irreplaceable\r\n"
    assert list(output.iterdir()) == [evidence]


def test_changed_implementation_source_preserves_an_incomplete_run(tmp_path, monkeypatch):
    source = write_config(tmp_path, tiny_contract())
    output = tmp_path / "source_changed"
    implementation = tmp_path / "declared_implementation.py"
    implementation.write_bytes(b"before\n")
    initial_hash = hashlib.sha256(implementation.read_bytes()).hexdigest()
    monkeypatch.setattr(maps, "_snapshot_sources", lambda *a: [
        {"path": str(implementation), "relative_path": implementation.name, "sha256": initial_hash}
    ])
    original = maps.integrate_map_point
    def changed(validated, initial):
        answer = original(validated, initial)
        implementation.write_bytes(b"after\n")
        return answer
    monkeypatch.setattr(maps, "integrate_map_point", changed)
    with pytest.raises(RuntimeError, match="Source files changed during the run"):
        maps.run_basin_maps(source, output)
    assert not (output / "manifest.json").exists()
    assert json.loads((output / ".hafo_run.json").read_text())["status"] == "in_progress"
    assert len(list((output / "trajectories").glob("*.npz"))) == 9


def test_terminal_window_rejects_early_equilibrium_and_last_point_capture():
    validated = maps.validate_map_contract(tiny_contract())
    times = np.arange(11) * .01
    states = np.zeros((11, 3))
    states[8:, 0] = [1., 1., 0.]
    result = maps.classify_map_trajectory(times, states, "ok", validated, {"E0": np.zeros(3)})
    assert result["class_label"] == "bounded_positive_mean"
    assert result["is_reference_target"] is False
    assert result["hidden_verified"] is False
    states[:] = 0.
    assert maps.classify_map_trajectory(times, states, "ok", validated, {"E0": np.zeros(3)})["class_id"] == 0


@pytest.mark.parametrize(("status", "class_id"), [("nonfinite", 5), ("diverged", 4), ("stopped_early", 4)])
def test_failure_threshold_incomplete_are_distinct(status, class_id):
    validated = maps.validate_map_contract(tiny_contract())
    result = maps.classify_map_trajectory(np.array([0., .01]), np.zeros((2, 3)), status, validated, {})
    assert result["class_id"] == class_id


@pytest.mark.parametrize("status", ["ok", "diverged", "diverged_early"])
def test_threshold_requires_an_observed_norm_crossing(status):
    validated = maps.validate_map_contract(tiny_contract())
    states = np.zeros((2, 3))
    states[-1, 0] = 121.
    result = maps.classify_map_trajectory(np.array([0., .01]), states, status, validated, {})
    assert result["class_id"] == 3
    assert result["classification_reason"] == "observed_norm_threshold_crossing"


@pytest.mark.parametrize("status", ["diverged", "diverged_early"])
def test_reported_divergence_without_observed_crossing_is_inconclusive(status):
    validated = maps.validate_map_contract(tiny_contract())
    # Equality with the threshold is not a crossing under the declared > rule.
    states = np.array([[0., 0., 0.], [120., 0., 0.]])
    result = maps.classify_map_trajectory(np.array([0., .01]), states, status, validated, {})
    assert result["class_id"] == 4
    assert result["classification_reason"] == "reported_divergence_without_observed_crossing"


def test_nonfinite_state_and_nonmonotone_time_are_not_terminal_equilibria():
    validated = maps.validate_map_contract(tiny_contract())
    times, states = np.arange(11) * .01, np.zeros((11, 3))
    states[5, 0] = np.nan
    assert maps.classify_map_trajectory(times, states, "ok", validated, {})["class_id"] == 5
    states[:] = 0
    times[5] = times[4]
    assert maps.classify_map_trajectory(times, states, "ok", validated, {"E0": np.zeros(3)})["class_id"] == 4


def test_explicit_caputo_diagnostic_has_own_dispatch(monkeypatch):
    config = tiny_contract()
    config["numerical_contract"].update(integrator="efork3", backend="caputo_diagnostic", memory_policy="full", memory_length=.1, caputo_history_accumulated=True)
    validated = maps.validate_map_contract(config)
    seen = {}
    from hidden_attractors.integrations import selector
    def caputo(rhs, initial, **options):
        seen.update(options)
        np.testing.assert_allclose(rhs(np.zeros(3)), 0)
        return np.array([0., .1]), np.zeros((2, 3)), "ok"
    monkeypatch.setattr(selector, "integrate", caputo)
    monkeypatch.setattr(maps, "adm_wu2023_integrate", lambda *a, **k: pytest.fail("Caputo silently dispatched to local ADM"))
    _, _, _, info = maps.integrate_map_point(validated, np.zeros(3))
    assert seen["integrator"] == "efork3" and seen["memory_mode"] == "full"
    assert seen["early_stop_config"] == {"enabled": False}
    assert seen["use_c_backend"] is False
    assert info["caputo_history_accumulated"] is True
    paths = {path.name for path in maps._implementation_paths(validated)}
    assert {"selector.py", "general.py", "efork.py", "_history.py", "_rhs.py"} <= paths
    assert "adm_wu2023.py" not in paths


def test_real_adm_three_by_three_map_replays_every_saved_trajectory(tmp_path):
    config = tiny_contract()
    source = write_config(tmp_path, config)
    output = tmp_path / "real_adm"
    manifest = maps.run_basin_maps(source, output)
    assert manifest["method"] == "adm_wu2023" and manifest["points_total"] == 9
    assert manifest["hidden_verified"] is False and manifest["reference_contact_enabled"] is False
    source_names = {Path(entry["path"]).name for entry in manifest["source_hashes"]}
    assert "adm_wu2023.py" in source_names and "efork.py" not in source_names
    for entry in manifest["source_hashes"]:
        assert hashlib.sha256((output / entry["snapshot_path"]).read_bytes()).hexdigest() == entry["sha256"]
    assert json.loads((output / ".hafo_run.json").read_text())["status"] == "completed"
    rows = list(csv.DictReader((output / "map_xy.csv").open(encoding="utf-8")))
    for row in rows:
        initial = np.array([float(row[key]) for key in ("x0", "y0", "z0")])
        direct_t, direct_x, direct_status, _ = adm_wu2023_integrate(config["parameters"], initial, q=.99, h=.01, N=10, divergence_norm=120)
        with np.load(output / row["trajectory_file"]) as saved:
            np.testing.assert_array_equal(saved["times"], direct_t)
            np.testing.assert_array_equal(saved["states"], direct_x)
            np.testing.assert_array_equal(saved["initial"], initial)
        assert row["integration_status"] == direct_status
    for entry in manifest["files"]:
        path = output / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    with np.load(output / "map_xy.npz") as matrix:
        assert matrix["class_ids"].shape == (3, 3)
        assert matrix["initial_conditions"].shape == (3, 3, 3)
    assert (output / "map_xy.png").stat().st_size > 10000


def test_standalone_cli_imports_local_package_and_rejects_contract_before_output(tmp_path):
    config = tiny_contract()
    config["numerical_contract"]["adm_order"] = 7
    source = write_config(tmp_path, config)
    output = tmp_path / "cli_invalid"
    script = ROOT / "validation/reference_cases/fractional_chua_arctan_wu2023/scripts/chua_arctan_wu2023_plot_basins.py"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(script), "--config", str(source), "--output-dir", str(output)],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 2
    assert "ADM requires order=4" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not output.exists()


def test_cli_horizon_change_requires_an_explicit_consistent_contract(tmp_path):
    source = write_config(tmp_path, tiny_contract())
    output = tmp_path / "cli_bad_horizon"
    script = ROOT / "validation/reference_cases/fractional_chua_arctan_wu2023/scripts/chua_arctan_wu2023_plot_basins.py"
    result = subprocess.run([sys.executable, str(script), "--config", str(source), "--output-dir", str(output), "--t-final", "1"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert "change h/N/T together" in result.stderr
    assert not output.exists()
