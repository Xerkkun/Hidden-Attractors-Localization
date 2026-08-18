from __future__ import annotations

import ctypes
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hidden_attractors.parallel as parallel_module
import hidden_attractors.integrations.efork as efork_module
from hidden_attractors.integrations.fractional_c import (
    GeneralFractionalCBackend,
    fractional_integrate,
)
from hidden_attractors.models.chua import ChuaParameters
from hidden_attractors.native.backends import (
    BasinBackend,
    FractionalChuaBackend,
    FractionalLyapunovBackend,
    FullHistoryABMBackend,
    GeneralFDEBackend,
    NativeFractionalVariationalBackend,
)
from hidden_attractors.native.contracts import FractionalLyapunovRequest
from hidden_attractors.native.rhs_registry import get_c_rhs_and_params
from hidden_attractors.parallel import compile_c_target, load_ctypes_library
from hidden_attractors.systems import get_system


def _decay(_time: float, state: np.ndarray) -> np.ndarray:
    return -np.asarray(state, dtype=np.float64)


def _fractional_kwargs() -> dict[str, object]:
    return {
        "rhs": _decay,
        "x0": np.array([1.0]),
        "q": 0.9,
        "h": 0.3,
        "t_final": 0.9,
        "method": "abm",
        "memory_mode": "full",
        "use_c_backend": False,
    }


def test_fractional_fixed_grid_rejects_overshoot_and_accepts_aligned_horizon() -> None:
    invalid = _fractional_kwargs()
    invalid["t_final"] = 1.0
    with pytest.raises(ValueError, match="integer number of fixed steps"):
        fractional_integrate(**invalid)

    times, states, status, info = fractional_integrate(**_fractional_kwargs())
    assert status == "ok"
    assert times[-1] == 0.9
    assert np.all(times <= 0.9)
    assert states.shape == (4, 1)
    assert info["effective_t_final"] == 0.9


def test_native_wrappers_reject_finite_huge_step_ratio_before_allocation() -> None:
    huge_horizon = 2_147_483_648.0
    invalid = _fractional_kwargs()
    invalid.update(h=1.0, t_final=huge_horizon, use_c_backend=True)
    with pytest.raises(ValueError, match="supported limit"):
        fractional_integrate(**invalid)

    with pytest.raises(ValueError, match="supported limit"):
        GeneralFDEBackend(lib=object()).integrate(
            _decay,
            np.array([1.0]),
            q=0.9,
            h=1.0,
            t_final=huge_horizon,
        )
    with pytest.raises(ValueError, match="supported limit"):
        FractionalChuaBackend(lib=object()).integrate_efork3(
            [0.1, 0.0, 0.0],
            q=0.9,
            h=1.0,
            Lm=1.0,
            t_final=huge_horizon,
        )
    with pytest.raises(ValueError, match="supported limit"):
        FullHistoryABMBackend(lib=object()).integrate(
            [0.1, 0.0, 0.0],
            q=0.9,
            h=1.0,
            t_final=huge_horizon,
        )
    with pytest.raises(ValueError, match="supported limit"):
        BasinBackend(lib=object()).classify_point(
            [0.1, 0.0, 0.0],
            q=0.9,
            h=1.0,
            Lm=1.0,
            t_final=huge_horizon,
            t_burn=0.0,
        )


@pytest.mark.parametrize("method", ["abm", "efork"])
@pytest.mark.parametrize("use_c_backend", [False, True])
def test_fractional_history_slice_keeps_t0_and_matches_sample_metadata(
    method: str,
    use_c_backend: bool,
) -> None:
    common = {
        **_fractional_kwargs(),
        "method": method,
        "use_c_backend": use_c_backend,
        "allow_python_fallback": False,
        "divergence_norm": None,
        "history_times": np.array([-0.6, -0.3, 0.0]),
        "history_states": np.ones((3, 1)),
    }
    without_history = fractional_integrate(**common, return_history=False)
    with_history = fractional_integrate(**common, return_history=True)

    times_new, states_new, status_new, info_new = without_history
    times_all, states_all, status_all, info_all = with_history
    assert status_new == status_all == "ok"
    np.testing.assert_allclose(times_new, np.array([0.0, 0.3, 0.6, 0.9]))
    np.testing.assert_allclose(times_all[:3], np.array([-0.6, -0.3, 0.0]))
    np.testing.assert_allclose(states_new, states_all[2:])
    assert info_new["n_steps"] == 3
    assert info_new["n_samples_returned"] == len(times_new) == 4
    assert info_all["n_samples_returned"] == len(times_all) == 6
    assert info_new["used_c_backend"] is use_c_backend


def test_public_efork_default_disabled_cutoff_uses_native_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fallback(**_kwargs: object) -> tuple[np.ndarray, np.ndarray, str]:
        raise AssertionError("Python EFORK fallback must not run")

    monkeypatch.setattr(efork_module, "_python_efork3_integrate", fail_fallback)
    times, states, status = efork_module.efork_integrate(
        get_system("chua-arctan"),
        np.array([0.1, 0.0, 0.0]),
        q=0.9,
        h=0.1,
        t_final=0.2,
        early_stop_config={"enabled": False},
    )
    assert status == "ok"
    np.testing.assert_allclose(times, np.array([0.0, 0.1, 0.2]))
    assert states.shape == (3, 3)


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"method": "unknown"}, "unknown method"),
        ({"memory_mode": "mystery"}, "exactly 'full' or 'window'"),
        ({"memory_mode": "window"}, "memory_window_length is required"),
        (
            {"memory_mode": "window", "memory_window_length": True},
            "positive integer",
        ),
        ({"early_stop_config": {"enabled": 2}}, "must be boolean or 0/1"),
    ],
)
def test_fractional_python_rejects_unknown_enums_and_window_values(
    updates: dict[str, object], error: str
) -> None:
    kwargs = _fractional_kwargs()
    kwargs.update(updates)
    with pytest.raises((TypeError, ValueError), match=error):
        fractional_integrate(**kwargs)


@pytest.mark.parametrize(
    ("history_times", "history_states", "error"),
    [
        (np.array([0.0]), None, "provided together"),
        (np.array([[0.0]]), np.array([[1.0]]), "history_times"),
        (np.array([-0.3, 0.0]), np.array([[1.0]]), "history_states"),
        (np.array([-0.3, np.nan]), np.array([[1.0], [1.0]]), "finite"),
        (
            np.array([-0.4, 0.0]),
            np.array([[1.0], [1.0]]),
            "same h grid",
        ),
        (
            np.array([-0.6, -0.3]),
            np.array([[1.0], [1.0]]),
            "end at t=0",
        ),
        (
            np.array([-0.3, 0.0]),
            np.array([[1.0], [2.0]]),
            "last history state",
        ),
    ],
)
def test_fractional_python_validates_complete_history_contract(
    history_times: np.ndarray,
    history_states: np.ndarray | None,
    error: str,
) -> None:
    kwargs = _fractional_kwargs()
    kwargs.update(history_times=history_times, history_states=history_states)
    with pytest.raises(ValueError, match=error):
        fractional_integrate(**kwargs)


@pytest.mark.parametrize(
    ("equilibria", "error"),
    [
        ([np.array([0.0, 1.0])], "shape"),
        ([np.array([np.nan])], "finite"),
    ],
)
def test_fractional_python_validates_equilibria(
    equilibria: list[np.ndarray], error: str
) -> None:
    kwargs = _fractional_kwargs()
    kwargs["equilibria"] = equilibria
    with pytest.raises(ValueError, match=error):
        fractional_integrate(**kwargs)


def _call_fractional_c(
    backend: GeneralFractionalCBackend,
    *,
    method: int = 0,
    memory_mode: int = 0,
    window: int = 0,
    t_final: float = 0.2,
    history_times: np.ndarray | None = None,
    history_states: np.ndarray | None = None,
    equilibria: np.ndarray | None = None,
    out_times_capacity: int | None = None,
) -> int:
    callback = backend.RHS_CALLBACK(
        lambda _t, x_ptr, dx_ptr, _n, _params: dx_ptr.__setitem__(0, -x_ptr[0])
    )
    x0 = np.array([1.0], dtype=np.float64)
    htimes = np.array([0.0], dtype=np.float64) if history_times is None else history_times
    hstates = np.array([1.0], dtype=np.float64) if history_states is None else history_states
    history_len = 0 if history_times is None else int(history_times.size)
    eq = np.array([0.0], dtype=np.float64) if equilibria is None else equilibria
    num_eq = 0 if equilibria is None else 1
    out_times = np.empty(3, dtype=np.float64)
    out_states = np.empty(3, dtype=np.float64)
    out_steps = ctypes.c_int()
    status = ctypes.c_int()
    return int(
        backend.lib.integrate_fractional_c(
            callback,
            ctypes.c_void_p(),
            1,
            x0,
            0.9,
            0.1,
            t_final,
            method,
            memory_mode,
            window,
            htimes,
            hstates,
            history_len,
            htimes.size,
            hstates.size,
            120.0,
            out_times,
            out_states,
            out_times.size if out_times_capacity is None else out_times_capacity,
            out_states.size,
            ctypes.byref(out_steps),
            ctypes.byref(status),
            0,
            0,
            80.0,
            5,
            1.25,
            0,
            1.0e-3,
            1.0e-4,
            200,
            5.0,
            eq,
            num_eq,
            eq.size if num_eq else 0,
        )
    )


def test_fractional_c_rejects_enums_grid_history_equilibria_and_capacity() -> None:
    backend = GeneralFractionalCBackend.get_instance()
    assert _call_fractional_c(backend, method=99) == -1
    assert _call_fractional_c(backend, memory_mode=99) == -1
    assert _call_fractional_c(backend, memory_mode=1, window=0) == -1
    assert _call_fractional_c(backend, t_final=0.25) == -1
    assert _call_fractional_c(backend, out_times_capacity=1) == -4
    assert (
        _call_fractional_c(
            backend,
            history_times=np.array([-0.2, 0.0], dtype=np.float64),
            history_states=np.array([1.0, 2.0], dtype=np.float64),
        )
        == -1
    )
    assert _call_fractional_c(backend, equilibria=np.array([np.nan])) == -1


def test_general_fde_rejects_unknown_method_and_noninteger_horizon() -> None:
    class NotCalled:
        pass

    backend = GeneralFDEBackend(lib=NotCalled())
    with pytest.raises(ValueError, match="integer number of fixed steps"):
        backend.integrate(_decay, np.array([1.0]), 0.9, 0.3, 1.0)
    with pytest.raises(ValueError, match="exactly 'efork' or 'abm'"):
        backend.integrate(
            _decay, np.array([1.0]), 0.9, 0.3, 0.9, integrator="unknown"
        )


def test_general_fde_c_rejects_small_output_capacity() -> None:
    backend = GeneralFDEBackend.build(output_name="general_fde_hardening_test")

    @backend.RHS_CALLBACK
    def callback(_time, state, derivative):
        derivative[0] = -state[0]

    out = np.empty(2, dtype=np.float64)
    rc = int(
        backend.lib.integrate_general_efork_c(
            callback, np.array([1.0]), 1, 0.9, 0.1, 0.2, 120.0, out, out.size
        )
    )
    assert rc == -4


def _variational_request() -> FractionalLyapunovRequest:
    return FractionalLyapunovRequest(
        system_id="lorenz",
        x0=np.array([1.0, 2.0, 3.0]),
        parameters={"sigma": 10.0, "beta": 8.0 / 3.0, "rho": 28.0},
        q=0.9,
        h=0.1,
        t_final=1.0,
        t_burn=0.2,
        reorthonormalization_time=0.2,
    )


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"q": np.nan}, "q must be finite"),
        ({"q": 1.0}, "0 < q < 1"),
        ({"t_final": 0.95}, "integer number of fixed steps"),
        ({"t_burn": 0.15}, "integer number of fixed steps"),
        ({"reorthonormalization_time": 0.15}, "integer number of fixed steps"),
        ({"fft_block_size": 0}, "positive C int"),
        ({"fft_block_size": True}, "positive integer"),
        ({"divergence_norm": -1.0}, "non-negative"),
        ({"x0": np.array([1.0, np.nan, 3.0])}, "finite"),
        (
            {"parameters": {"sigma": np.inf, "beta": 2.0, "rho": 28.0}},
            "finite",
        ),
        (
            {
                "h": 1.0,
                "t_final": 2_147_483_648.0,
                "t_burn": 0.0,
                "reorthonormalization_time": 1.0,
            },
            "supported limit",
        ),
    ],
)
def test_variational_backend_validates_grid_finiteness_and_capacity(
    updates: dict[str, object], error: str
) -> None:
    backend = NativeFractionalVariationalBackend(lib=object(), build_metadata={})
    request = replace(_variational_request(), **updates)
    with pytest.raises((TypeError, ValueError), match=error):
        backend.run(request)


def test_content_addressed_build_is_atomic_and_tracks_local_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    header = tmp_path / "value.h"
    source = tmp_path / "kernel.c"
    header.write_text("#define VALUE 7\n", encoding="utf-8")
    source.write_text(
        '#include "value.h"\nint native_value(void) { return VALUE; }\n',
        encoding="utf-8",
    )
    requested = tmp_path / ("kernel.dll" if __import__("sys").platform == "win32" else "kernel.so")

    def build() -> Path:
        return compile_c_target(
            source, requested, target_kind="shared", openmp=False
        ).path

    with ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(lambda _index: build(), range(12)))
    assert len(set(paths)) == 1
    first = paths[0]
    assert first.is_file()
    assert not list(tmp_path.glob("*.tmp*"))

    header.write_text("#define VALUE 8\n", encoding="utf-8")
    second = build()
    assert second.is_file()
    assert second != first

    monkeypatch.setattr(parallel_module.platform, "machine", lambda: "simulated-arch")
    simulated_arch, _command = parallel_module._content_addressed_output(
        source,
        requested,
        target_kind="shared",
        openmp=False,
    )
    assert simulated_arch != second


def test_native_loader_rebuilds_one_poisoned_content_addressed_artifact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "healthy.c"
    source.write_text("int native_value(void) { return 17; }\n", encoding="utf-8")
    suffix = ".dll" if __import__("sys").platform == "win32" else ".so"
    requested = tmp_path / f"healthy{suffix}"
    artifact = compile_c_target(
        source, requested, target_kind="shared", openmp=False
    ).path
    sidecar = artifact.with_name(artifact.name + ".sha256")
    assert sidecar.is_file()
    artifact.write_bytes(b"not a shared library")

    result, library = load_ctypes_library(
        source,
        requested,
        expected_symbols=("native_value",),
        openmp=False,
    )
    library.native_value.argtypes = []
    library.native_value.restype = ctypes.c_int
    assert result.path == artifact
    assert sidecar.read_text(encoding="ascii").strip() == parallel_module._sha256_file(artifact)
    assert library.native_value() == 17


def test_chua_global_parameters_are_transactional_across_threads() -> None:
    backend = FractionalChuaBackend.build(output_name="chua_thread_hardening_test")
    params_a = ChuaParameters(alpha=8.4562)
    params_b = ChuaParameters(alpha=18.0)
    kwargs = dict(q=0.9, h=0.05, Lm=0.1, t_final=0.1)
    seed = np.array([0.2, -0.1, 0.3])

    backend.set_nonsmooth_params(params_a)
    expected_a = backend.integrate_efork3(seed, **kwargs)
    backend.set_nonsmooth_params(params_b)
    expected_b = backend.integrate_efork3(seed, **kwargs)
    barrier = threading.Barrier(2)

    def run(params: ChuaParameters) -> np.ndarray:
        backend.set_nonsmooth_params(params)
        barrier.wait(timeout=10.0)
        return backend.integrate_efork3(seed, **kwargs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        result_a, result_b = list(pool.map(run, (params_a, params_b)))
    np.testing.assert_allclose(result_a, expected_a, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result_b, expected_b, rtol=0.0, atol=0.0)


def test_abm_global_parameters_are_transactional_across_threads() -> None:
    backend = FullHistoryABMBackend.build(output_name="chua_abm_thread_hardening_test")
    params_a = ChuaParameters(alpha=8.4562)
    params_b = ChuaParameters(alpha=18.0)
    kwargs = dict(q=0.9, h=0.05, t_final=0.1)
    seed = np.array([0.2, -0.1, 0.3])

    backend.set_nonsmooth_params(params_a)
    expected_a = backend.integrate(seed, **kwargs)
    backend.set_nonsmooth_params(params_b)
    expected_b = backend.integrate(seed, **kwargs)
    barrier = threading.Barrier(2)

    def run(params: ChuaParameters) -> np.ndarray:
        backend.set_nonsmooth_params(params)
        barrier.wait(timeout=10.0)
        return backend.integrate(seed, **kwargs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        result_a, result_b = list(pool.map(run, (params_a, params_b)))
    np.testing.assert_allclose(result_a, expected_a, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result_b, expected_b, rtol=0.0, atol=0.0)


def test_basin_global_parameters_are_transactional_across_threads() -> None:
    backend = BasinBackend.build(output_name="chua_basin_thread_hardening_test")
    params_a = ChuaParameters(alpha=8.4562)
    params_b = ChuaParameters(alpha=18.0)

    backend.set_nonsmooth_params(params_a)
    expected_a = backend.equilibria()
    backend.set_nonsmooth_params(params_b)
    expected_b = backend.equilibria()
    barrier = threading.Barrier(2)

    def run(params: ChuaParameters) -> dict[str, np.ndarray]:
        backend.set_nonsmooth_params(params)
        barrier.wait(timeout=10.0)
        return backend.equilibria()

    with ThreadPoolExecutor(max_workers=2) as pool:
        result_a, result_b = list(pool.map(run, (params_a, params_b)))
    for label in expected_a:
        np.testing.assert_allclose(result_a[label], expected_a[label], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result_b[label], expected_b[label], rtol=0.0, atol=0.0)


def test_rhs_registry_requires_exact_identity_and_finite_parameters() -> None:
    backend = GeneralFractionalCBackend.get_instance()
    similar_name = SimpleNamespace(name="unregistered-arctan-experiment")
    assert get_c_rhs_and_params(similar_name, backend.lib) == (None, None)

    invalid = SimpleNamespace(
        system_id="chua_fractional_arctan",
        alpha=float("inf"),
    )
    with pytest.raises(ValueError, match="must be finite"):
        get_c_rhs_and_params(invalid, backend.lib)


def test_chua_single_trajectory_abis_reject_small_capacity() -> None:
    efork = FractionalChuaBackend.build(output_name="chua_capacity_hardening_test")
    out = np.empty(8, dtype=np.float64)
    assert (
        int(
            efork.lib.integrate_chua_efork3(
                0.1, 0.0, 0.0, 0.9, 0.1, 0.1, 0.2, 0.0, 1.0, out, 1
            )
        )
        == -14
    )

    abm = FullHistoryABMBackend.build(output_name="chua_abm_capacity_hardening_test")
    assert (
        int(
            abm.lib.integrate_chua_abm_full_history(
                0.1, 0.0, 0.0, 0.9, 0.1, 0.2, out, 1
            )
        )
        == -4
    )


def test_lyapunov_wrapper_and_executable_reject_invalid_inputs(tmp_path: Path) -> None:
    executable_backend = FractionalLyapunovBackend.build(
        output_name="chua_lyapunov_input_hardening_test"
    )
    with pytest.raises(ValueError, match="x0"):
        executable_backend.run(
            [1.0, 2.0],
            q=0.9,
            h=0.1,
            Lm=0.2,
            t_burn=0.0,
            n_blocks=1,
            t_block=0.1,
            convergence_csv=tmp_path / "invalid.csv",
        )
    with pytest.raises(ValueError, match="integer number of fixed steps"):
        executable_backend.run(
            [1.0, 0.0, 0.0],
            q=0.9,
            h=0.3,
            Lm=0.6,
            t_burn=0.0,
            n_blocks=1,
            t_block=1.0,
            convergence_csv=tmp_path / "invalid.csv",
        )

    command = [str(executable_backend.executable), *("nan" for _ in range(14))]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode != 0


def test_hidden_backend_executable_rejects_nonfinite_and_malformed_values(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "hidden_attractors"
        / "native"
        / "csrc"
        / "chua_hidden_backend.c"
    )
    suffix = ".exe" if __import__("sys").platform == "win32" else ""
    executable = compile_c_target(
        source,
        tmp_path / f"chua_hidden_input_hardening{suffix}",
        target_kind="executable",
        openmp=True,
    ).path

    for invalid_value in ("nan", "1e", "inf"):
        completed = subprocess.run(
            [str(executable), "--alpha_chua", invalid_value],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode != 0
        assert "inv" in completed.stderr.lower()


def test_orphan_native_grid_and_bifurcation_symbols_are_not_exported() -> None:
    fractional = FractionalChuaBackend.build(output_name="chua_exports_hardening_test")
    assert not hasattr(fractional.lib, "compute_bifurcation_sweep_efork3")
    assert not hasattr(fractional.lib, "set_frac_backend_workers")

    basin = BasinBackend.build(output_name="chua_basin_exports_hardening_test")
    assert not hasattr(basin.lib, "compute_basin_xy")
    assert not hasattr(basin.lib, "compute_basin_plane")
    assert not hasattr(basin.lib, "set_basin_workers")
    assert not hasattr(basin.lib, "get_chua_params")
