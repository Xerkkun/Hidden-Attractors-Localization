from __future__ import annotations

import json
import numpy as np
import pytest

from hidden_attractors.verification import hiddenness, sphere_tests, basins
from hidden_attractors.verification.terminal_policy import PROBE_POLICY, completion_tail
from hidden_attractors.verification.hiddenness_contract import verify_hiddenness_contract


class LinearSystem:
    parameters = {"q": 1.0}
    def evaluate(self, state):
        return -np.asarray(state)


def run(monkeypatch, times=None, states=None, status="ok", **kwargs):
    seen = {}
    if times is None:
        times = np.linspace(0, 1, 11)
    if states is None:
        states = np.ones((len(times), 3))
    def integrate(**arguments):
        seen.update(arguments)
        return times, states, status
    monkeypatch.setattr(hiddenness, "integrate_general", integrate)
    values = dict(system=LinearSystem(), x0=np.ones(3), transfer_mode="integer",
                  integrator="rk4", t_final=1., t_burn=.5, h=.1,
                  ref_tail=np.ones((10,3)), stable_equilibria=[np.zeros(3)],
                  equilibrium_tol=.1, divergence_norm=10., q_dynamics_effective=1.)
    values.update(kwargs)
    return hiddenness.run_neighborhood_probe(**values), seen


def test_early_capture_is_disabled_and_later_threshold_wins(monkeypatch):
    states = np.zeros((11,3)); states[8:,0] = 20
    config = {"enabled": True, "equilibrium": {"enabled": True}, "divergence_norm": .01}
    result, seen = run(monkeypatch, states=states, early_stop_config=config)
    assert result["destination"] == "threshold_crossed"
    assert result["diagnostics"]["first_threshold_time"] == pytest.approx(.8)
    assert seen["early_stop_config"]["enabled"] is False
    assert seen["early_stop_config"]["equilibrium"]["enabled"] is False
    assert config["enabled"] is True  # no mutation of the declared contract


@pytest.mark.parametrize("status", ["converged_equilibrium_early", "diverged_early", "unexpected"])
def test_early_or_unknown_status_is_inconclusive_without_observed_crossing(monkeypatch,status):
    result,_ = run(monkeypatch, status=status, times=np.linspace(0,.4,5), states=np.zeros((5,3)))
    assert result["destination"] == "inconclusive"


@pytest.mark.parametrize("times, expected", [
    (np.linspace(0,.4,5), "incomplete_horizon"),
    (np.array([0,.8,.9,1.]), "insufficient_terminal_samples"),
    (np.array([0,.8,.8,1.]), "nonmonotone_time"),
])
def test_physical_horizon_and_tail_are_required(monkeypatch,times,expected):
    result,_ = run(monkeypatch,times=times,states=np.zeros((len(times),3)),t_burn=.85)
    assert result["destination"] == "inconclusive"
    assert result["diagnostics"]["reason"] == expected


def test_last_point_near_equilibrium_does_not_capture_wide_tail(monkeypatch):
    states = np.ones((11,3)); states[-1] = 0
    result,_ = run(monkeypatch,states=states,ref_tail=np.full((5,3),20.))
    assert result["destination"] == "other_finite"


def test_q95_distance_is_required_even_if_tail_span_passes(monkeypatch):
    states = np.zeros((11,3));states[5:10,0] = .15
    result,_ = run(monkeypatch,states=states,ref_tail=np.full((5,3),20.),equilibrium_span_tol=.2)
    assert result["destination"] == "other_finite"


def test_equilibrium_span_is_required(monkeypatch):
    states = np.zeros((11,3));states[5:,0] = [.09,-.09,.09,-.09,.09,0]
    result,_ = run(monkeypatch,states=states,ref_tail=np.full((5,3),20.),equilibrium_span_tol=.1)
    assert result["destination"] == "other_finite"


def test_terminal_equilibrium_has_name_diagnostics_and_actual_times(monkeypatch):
    result,_ = run(monkeypatch,states=np.zeros((11,3)),equilibria_dict={"E0":np.zeros(3)})
    assert result["destination"] == "equilibrium_terminal"
    assert result["equilibrium_name"] == "E0"
    assert result["diagnostics"]["tail_samples"] == 6
    assert result["diagnostics"]["equilibrium_q95_distance"] == 0
    assert np.array_equal(result["times"],np.linspace(0,1,11))


def test_irregular_times_select_tail_without_index_burn(monkeypatch):
    times = np.array([0.,.1,.2,.7,.9,1.])
    states = np.ones((6,3));states[:3] = 20
    result,_ = run(monkeypatch,times=times,states=states,divergence_norm=100)
    assert result["destination"] == "reference_contact"
    assert result["diagnostics"]["tail_samples"] == 3
    assert result["diagnostics"]["tail_start"] == .7


@pytest.mark.parametrize("state", [np.nan,np.inf])
def test_nonfinite_is_numerical_failure_not_threshold(monkeypatch,state):
    states = np.zeros((11,3));states[-1,0] = state
    result,_ = run(monkeypatch,states=states,status="diverged")
    assert result["destination"] == "numerical_failure"


def test_solver_failure_is_distinct(monkeypatch):
    result,_ = run(monkeypatch,status="solver_exception:bad")
    assert result["destination"] == "numerical_failure"


def test_explicit_legacy_reproduces_early_capture(monkeypatch):
    result,_ = run(monkeypatch,status="converged_equilibrium_early",states=np.zeros((11,3)),classification_policy="legacy_v1")
    assert result["destination"] == "stable_equilibrium"
    assert result["classification_policy"] == "legacy_v1"


def test_sphere_and_ball_are_different_reproducible_measures():
    sphere = hiddenness.generate_neighborhood_points(np.zeros(3),2,4000,"sphere_random",91)
    ball = hiddenness.generate_neighborhood_points(np.zeros(3),2,4000,"ball_random",91)
    assert np.allclose(np.linalg.norm(sphere,axis=1),2)
    radii = np.linalg.norm(ball,axis=1)/2
    assert np.all(radii <= 1)
    assert abs(np.mean(radii**3)-.5) < .02
    assert np.array_equal(ball,hiddenness.generate_neighborhood_points(np.zeros(3),2,4000,"ball_random",91))


def test_verification_slice_consumer_preserves_inconclusive_and_diagnostics(monkeypatch):
    monkeypatch.setattr(basins,"run_neighborhood_probe", lambda **kw: {
        "destination":"inconclusive","status":"ok","diagnostics":{"reason":"incomplete_horizon"}})
    payload = (0,1,np.zeros(3),LinearSystem(),"integer","rk4",1.,.5,.1,
               np.zeros((3,3)),[],.1,10.,"centroid_distance",.1,"integer","full",None,None,None,1.)
    i,j,code,diagnostics = basins._classify_point_worker(payload)
    assert (i,j,code) == (0,1,5)
    assert diagnostics["reason"] == "incomplete_horizon"


def config(mode="sphere_random"):
    return {"system_id":"linear","transfer_mode":"integer","integrator":"rk4",
            "equilibrium_tol":.1,"divergence_norm":10.,"target_match_metric":"centroid_distance",
            "target_match_tol":.1,"dynamics_mode":"integer","memory_mode":"full",
            "memory_window_length":None,"plot_sphere_tests":False,
            "sphere_tests":{"radii":[.2],"samples_initial":2,"samples_growth_factor":1.,
                            "directions_mode":mode,"random_seed":3,"t_final":1.,"t_burn":.5,"h":.1}}


def test_sphere_consumer_persists_complete_trace_and_truthful_sampling(monkeypatch,tmp_path):
    def fake(**kw):
        return {"destination":"inconclusive","status":"ok", "trajectory":np.ones((4,3)),
                "times":np.array([0.,.2,.3,.4]),"classification_policy":PROBE_POLICY,
                "diagnostics":{"reason":"incomplete_horizon","tail_samples":0}}
    monkeypatch.setattr(sphere_tests,"run_neighborhood_probe",fake)
    result = sphere_tests.run_sphere_probe_sweep(LinearSystem(),config(),{"E0":np.zeros(3)},[],np.ones((5,3)),str(tmp_path/"new"),q_dynamics_effective=1.)
    assert result["metadata"]["sampling_domain"] == "sphere_surface"
    assert result["summary_records"][0]["INCONCLUSIVE"] == 2
    assert result["summary_records"][0]["criterio_local"] == "INCONCLUSIVE"
    record = result["probe_runs"][0]
    stored = np.load(tmp_path/"new"/record["trajectory_file"])
    assert np.array_equal(stored["times"],[0,.2,.3,.4])
    assert np.array_equal(stored["states"],np.ones((4,3)))
    assert (tmp_path/"new/sphere_tests_metadata.json").exists()
    with pytest.raises(FileExistsError):
        sphere_tests.run_sphere_probe_sweep(LinearSystem(),config(),{"E0":np.zeros(3)},[],np.ones((5,3)),str(tmp_path/"new"),q_dynamics_effective=1.)


def test_axis_sampling_records_actual_count_without_none_entries(monkeypatch,tmp_path):
    cfg = config("coordinate_axes");cfg["sphere_tests"]["samples_initial"] = 10
    monkeypatch.setattr(sphere_tests,"run_neighborhood_probe",lambda **kw:{
        "destination":"other_finite","status":"ok","trajectory":np.ones((11,3)),
        "times":np.linspace(0,1,11),"classification_policy":PROBE_POLICY,
        "diagnostics":{"tail_samples":6}})
    result=sphere_tests.run_sphere_probe_sweep(LinearSystem(),cfg,{"E0":np.zeros(3)},[],np.ones((5,3)),str(tmp_path/"axes"),q_dynamics_effective=1.)
    assert len(result["probe_runs"]) == 6
    assert result["summary_records"][0]["samples"] == 6
    assert result["summary_records"][0]["requested_samples"] == 10


def test_sweep_uses_contract_metric_and_serializes_nonfinite_as_null(monkeypatch,tmp_path):
    seen = []
    def fake(**kw):
        seen.append(kw)
        states=np.ones((11,3));states[-1,0]=np.nan
        return {"destination":"numerical_failure","status":"nonfinite_solution",
                "trajectory":states,"times":np.linspace(0,1,11),"classification_policy":PROBE_POLICY,
                "diagnostics":{"reason":"nonfinite_solution","max_norm":np.inf,"tail_samples":0}}
    cfg=config();cfg["hiddenness"]={"target_match_metric":"nn_percentile","target_match_tol":.07,"target_match_nn_percentile":95.}
    monkeypatch.setattr(sphere_tests,"run_neighborhood_probe",fake)
    result=sphere_tests.run_sphere_probe_sweep(LinearSystem(),cfg,{"E0":np.zeros(3)},[],np.ones((5,3)),str(tmp_path/"failed"),q_dynamics_effective=1.)
    assert all(call["target_match_metric"] == "nn_percentile" for call in seen)
    assert all(call["target_match_tol"] == .07 for call in seen)
    assert result["metadata"]["reference_contact_metric"] == "nn_percentile"
    line=(tmp_path/"failed/sphere_tests_runs.jsonl").read_text().splitlines()[0]
    record=json.loads(line,parse_constant=lambda value: pytest.fail("Nonstandard JSON constant: "+value))
    assert record["final_state"][0] is None
    assert record["diagnostics"]["max_norm"] is None


def test_contract_counts_operational_contacts_and_blocks_unknown_destinations():
    values=dict(equilibria={"E0":np.zeros(3)},sphere_summary_records=[],required_radii=[.1])
    report=verify_hiddenness_contract(**values,probe_runs=[{"equilibrium":"E0","radius":.1,
        "destination":"reference_contact","classification_policy":PROBE_POLICY,
        "diagnostics":{"tail_samples":300}}])
    assert report["target_hits_total"] == 1
    assert report["reference_contact_detected"] is True
    assert report["self_excited_contact_detected"] is False
    assert report["promotion_verdict"] == "reference_contact_detected"
    report=verify_hiddenness_contract(**values,probe_runs=[{"equilibrium":"E0","radius":.1,"destination":"inconclusive"}])
    assert report["inconclusive_total"] == 1
    assert report["protocol_complete"] is False
