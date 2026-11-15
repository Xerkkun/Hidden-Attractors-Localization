"""Recorded scientific output must survive accidental reuse and competing runs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import pytest

from hidden_attractors._run_output import (
    RUN_MARKER, RunOutputClaim, claim_output_directory,
    complete_output_claim, require_output_claim,
)


def test_new_run_can_be_completed_but_not_reopened(tmp_path):
    output = tmp_path / "new_run"
    claim = claim_output_directory(output, run_kind="test")
    assert require_output_claim(output, claim) == output
    (output / "trajectory.csv").write_text("t,x\n0,1\n")
    complete_output_claim(claim)
    assert json.loads((output / RUN_MARKER).read_text())["status"] == "completed"
    with pytest.raises(ValueError, match="active run"):
        require_output_claim(output, claim)
    with pytest.raises(FileExistsError):
        claim_output_directory(output, run_kind="test")


@pytest.mark.parametrize("existing", ["basin_data.csv", "reference_attractor.csv", "hiddenness_summary.json", "unknown.bin", "empty_child/"])
def test_existing_evidence_is_untouched(tmp_path, existing):
    artifact = tmp_path / existing
    if existing.endswith("/"):
        artifact.mkdir()
        before = None
    else:
        artifact.write_bytes(b"immutable evidence\x00\xff")
        before = hashlib.sha256(artifact.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="not empty"):
        claim_output_directory(tmp_path, run_kind="new")
    assert not (tmp_path / RUN_MARKER).exists()
    if before is not None:
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == before


def test_claim_cannot_be_replaced_by_boolean_or_other_run(tmp_path):
    claim = claim_output_directory(tmp_path / "one", run_kind="test")
    with pytest.raises(TypeError, match="not an overwrite flag"):
        require_output_claim(claim.path, True)
    with pytest.raises(ValueError, match="different directory"):
        require_output_claim(tmp_path / "two", claim)
    forged = RunOutputClaim(claim.path, "other-run", "test")
    with pytest.raises(ValueError, match="active run"):
        require_output_claim(claim.path, forged)


def test_competing_writers_only_one_claims_output(tmp_path):
    def enter(_):
        try:
            return claim_output_directory(tmp_path / "competing", run_kind="test")
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=4) as executor:
        claims = list(executor.map(enter, range(4)))
    assert sum(c is not None for c in claims) == 1


def test_basin_rejects_used_output_before_calling_solver(tmp_path, monkeypatch):
    from hidden_attractors.workflows import basin_runner
    old = tmp_path / "basin_data.csv"
    old.write_bytes(b"previous grid\n")
    solver_calls = []
    monkeypatch.setattr(basin_runner, "integrate", lambda **kw: solver_calls.append(kw))
    with pytest.raises(FileExistsError, match="not empty"):
        basin_runner.run_basin_workflow({"run_basin_slices": True, "output_dir": str(tmp_path)})
    assert not solver_calls
    assert old.read_bytes() == b"previous grid\n"
    assert list(tmp_path.iterdir()) == [old]


def test_failed_basin_keeps_claim_and_prevents_a_second_integration(tmp_path, monkeypatch):
    from hidden_attractors.workflows import basin_runner
    import numpy as np

    class System:
        parameters = {"q": 1.}
        rhs = staticmethod(lambda t, x: np.zeros(3))

        def equilibrium_points(self):
            return {"E0": np.zeros(3)}

    calls = []
    def fail(**kw):
        calls.append(kw)
        raise RuntimeError("injected reference solver failure")
    monkeypatch.setattr(basin_runner, "get_system", lambda _: System())
    monkeypatch.setattr(basin_runner, "integrate", fail)
    config = {"run_basin_slices": True, "output_dir": str(tmp_path), "q": 1., "integrator": "rk4"}
    with pytest.raises(RuntimeError, match="injected reference"):
        basin_runner.run_basin_workflow(config)
    marker = (tmp_path / RUN_MARKER).read_bytes()
    assert json.loads(marker)["status"] == "in_progress"
    with pytest.raises(FileExistsError):
        basin_runner.run_basin_workflow(config)
    assert len(calls) == 1
    assert (tmp_path / RUN_MARKER).read_bytes() == marker


def test_simple_dispatch_never_writes_before_the_workflow_claim(tmp_path, monkeypatch):
    from hidden_attractors.workflows import simple_runner
    observed = []
    def centered(config):
        observed.extend(tmp_path.iterdir())
        return {"delegated": True}
    monkeypatch.setattr(simple_runner, "run_centered_lure_df_workflow", centered)
    result = simple_runner.run_simple_workflow({"output_dir": str(tmp_path)})
    assert result == {"delegated": True}
    assert observed == [] and not list(tmp_path.iterdir())
