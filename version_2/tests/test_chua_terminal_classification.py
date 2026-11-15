"""Finite equilibrium windows must survive observation of the complete horizon."""
from __future__ import annotations

import ctypes
from pathlib import Path
import sys

import numpy as np
import pytest

from hidden_attractors.basins.classification import class_label
from hidden_attractors.native.backends import BasinBackend
from hidden_attractors.parallel import compile_c_target


@pytest.fixture(scope="module")
def observed_classifier(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    csrc = root / "hidden_attractors/native/csrc"
    directory = tmp_path_factory.mktemp("terminal_chua")
    wrapper = directory / "observed.c"
    wrapper.write_text(
        '#define main hidden_backend_cli_main\n'
        f'#include "{(csrc / "chua_hidden_backend.c").as_posix()}"\n'
        '#undef main\n'
        '#if defined(_WIN32)\n#define EXPORT __declspec(dllexport)\n'
        '#else\n#define EXPORT __attribute__((visibility("default")))\n#endif\n'
        'EXPORT int observe(const double *states,int n,const double *eq,int cap,int policy,int *diagnostic){\n'
        'EqSet equilibria={0};equilibria.n_eq=1;memcpy(equilibria.val[0],eq,3*sizeof(double));\n'
        'ThresholdCfg threshold={0};threshold.R_DIV=80;threshold.EPS_EQ=.03;\n'
        'threshold.CAP_WIN=cap;threshold.CLASSIFIER_POLICY=policy;\n'
        'HafoEquilibriumObservation observation;\n'
        'int result=classify_equilibrium_or_divergence_observed(states,n,&equilibria,&threshold,&observation);\n'
        'diagnostic[0]=observation.visited;diagnostic[1]=observation.terminal_equilibrium;\n'
        'diagnostic[2]=observation.first_threshold_index;diagnostic[3]=observation.first_nonfinite_index;\n'
        'return result;}\n', encoding="utf-8"
    )
    suffix = ".dll" if sys.platform == "win32" else ".so"
    compiled = compile_c_target(wrapper, directory / ("observed" + suffix), target_kind="shared", openmp=False)
    library = ctypes.CDLL(str(compiled.path))
    function = library.observe
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integers = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    function.argtypes = [array, ctypes.c_int, array, ctypes.c_int, ctypes.c_int, integers]
    function.restype = ctypes.c_int

    def observe(states, *, cap=90, policy=1, equilibrium=(0., 0., 0.)):
        states = np.ascontiguousarray(states, dtype=np.float64)
        diagnostic = np.empty(4, dtype=np.int32)
        result = function(states, len(states), np.asarray(equilibrium, dtype=np.float64), cap, policy, diagnostic)
        return result, diagnostic.tolist()
    return observe


def test_stable_linear_decay_has_a_terminal_window(observed_classifier):
    times = np.linspace(0., 10., 1001)
    states = np.column_stack([np.exp(-times), np.zeros((len(times), 2))])
    result, diagnostic = observed_classifier(states)
    assert result == 0
    assert diagnostic == [1, 0, -1, -1]


def test_unstable_linear_departure_overrides_early_capture(observed_classifier):
    times = np.linspace(0., 20., 2001)
    states = np.column_stack([1.e-5*np.exp(times), np.zeros((len(times), 2))])
    assert observed_classifier(states, policy=0)[0] == 0
    result, diagnostic = observed_classifier(states)
    assert result == 1  # observed norm threshold, not a proof of divergence
    assert diagnostic[0] == 1 and diagnostic[1] == -1
    assert diagnostic[2] == int(np.flatnonzero(states[:, 0] > 80)[0])


def test_qualified_visit_is_separate_from_terminal_proximity(observed_classifier):
    states = np.zeros((201, 3))
    states[100:, 0] = 2.
    result, diagnostic = observed_classifier(states)
    assert result == -1
    assert diagnostic == [1, -1, -1, -1]


def test_short_and_finite_unresolved_observations_are_not_equilibrium(observed_classifier):
    assert observed_classifier(np.zeros((89, 3)))[0] == -1
    assert observed_classifier(np.ones((200, 3)))[0] == -1


def test_nonfinite_tail_is_failure_instead_of_equilibrium(observed_classifier):
    states = np.zeros((201, 3))
    states[-1, 0] = np.nan
    assert observed_classifier(states, policy=0)[0] == 0
    result, diagnostic = observed_classifier(states)
    assert result == 6
    assert diagnostic[3] == 200


@pytest.fixture(scope="module")
def basin_backend():
    return BasinBackend.build(output_name="chua_terminal_classification_test")


def _unstable_seed(backend):
    equilibrium = backend.equilibria()["E+"]
    jacobian = np.array([[-8.4562*(1.-1.1468), 8.4562, 0.], [1., -1., 1.], [0., -12.0732, -.0052]])
    values, vectors = np.linalg.eig(jacobian)
    direction = vectors[:, np.argmax(values.real)].real
    direction /= np.linalg.norm(direction)
    if direction[0] < 0:
        direction *= -1
    return equilibrium + 1.e-5*direction


def test_native_basin_c01_late_escape_is_not_equilibrium(basin_backend):
    kwargs = dict(q=1., h=.01, Lm=8., t_final=8., t_burn=0.,
                  equilibrium_tol=.03, cap_win=90, divergence_norm=80.)
    seed = _unstable_seed(basin_backend)
    assert basin_backend.classify_point(seed, classifier_policy="legacy_v1", **kwargs) == 0
    assert basin_backend.classify_point(seed, **kwargs) == 3


def test_native_basin_c01_visit_without_escape_is_separate(basin_backend):
    kwargs = dict(q=1., h=.01, Lm=5., t_final=5., t_burn=0.,
                  equilibrium_tol=.03, cap_win=90, divergence_norm=80.)
    assert basin_backend.classify_point(_unstable_seed(basin_backend), **kwargs) == 6


def test_native_basin_equilibrium_and_unresolved_horizons(basin_backend):
    kwargs = dict(q=1., h=.01, Lm=1., t_final=1., t_burn=0., cap_win=90)
    assert basin_backend.classify_point([0., 0., 0.], **kwargs) == 0
    assert basin_backend.classify_point([0., 0., 0.], **{**kwargs, "cap_win": 102}) == 4


def test_labels_distinguish_finite_evidence_and_keep_legacy_text():
    assert class_label(0) == "equilibrium"
    assert class_label(0, classifier_policy="terminal_v2") == "equilibrium_terminal"
    assert class_label(3, classifier_policy="terminal_v2") == "threshold_crossed"
    assert class_label(6, classifier_policy="terminal_v2") == "equilibrium_visit"
    with pytest.raises(ValueError, match="classifier_policy"):
        BasinBackend(lib=object()).classify_point([0., 0., 0.], q=1., h=.1,
            Lm=1., t_final=1., t_burn=0., classifier_policy="unversioned")
