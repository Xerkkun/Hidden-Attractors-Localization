"""Tests for scalar-time-series Lyapunov diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hidden_attractors.analysis.time_series_lyapunov as module
from hidden_attractors import EXPERIMENTAL, get_tier
from hidden_attractors.analysis import (
    TimeSeriesLyapunovResult,
    estimate_time_series_lyapunov,
    kaplan_yorke_dimension,
)


def test_kaplan_yorke_standard_spectrum() -> None:
    assert kaplan_yorke_dimension([0.4, 0.0, -2.0]) == pytest.approx(2.2)


def test_kaplan_yorke_sorts_and_handles_contracting_spectrum() -> None:
    assert kaplan_yorke_dimension([-3.0, 0.5, 0.0]) == pytest.approx(
        2.0 + 0.5 / 3.0
    )
    assert kaplan_yorke_dimension([-1.0, -2.0, -3.0]) == 0.0


def test_kaplan_yorke_caps_nonnegative_total_at_dimension() -> None:
    assert kaplan_yorke_dimension([0.3, 0.1, -0.2]) == 3.0


@pytest.mark.parametrize(
    "exponents",
    [[], [np.nan, 0.0], [[1.0, -1.0]]],
)
def test_kaplan_yorke_rejects_invalid_input(exponents: object) -> None:
    with pytest.raises(ValueError):
        kaplan_yorke_dimension(exponents)  # type: ignore[arg-type]


def test_estimator_passes_units_parameters_and_builds_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, dict[str, object]] = {}

    def fake_lyap_r(signal: np.ndarray, **kwargs: object) -> float:
        calls["rosenstein"] = {"n": signal.size, **kwargs}
        ks = np.arange(6, dtype=float)
        divergence = 1.0 + 0.008 * ks
        return 0.4, (ks, divergence, np.array([0.008, 1.0]))

    def fake_lyap_e(signal: np.ndarray, **kwargs: object) -> np.ndarray:
        calls["eckmann"] = {"n": signal.size, **kwargs}
        return np.array([0.4, 0.0, -2.0])

    fake_nolds = SimpleNamespace(
        lyap_r=fake_lyap_r,
        lyap_e=fake_lyap_e,
        lyap_r_len=lambda **kwargs: 1,
        lyap_e_len=lambda **kwargs: 1,
    )
    monkeypatch.setattr(module, "require_external", lambda name: fake_nolds)
    monkeypatch.setattr(module, "_backend_version", lambda: "test")

    signal = np.sin(np.linspace(0.0, 20.0, 512))
    result = estimate_time_series_lyapunov(
        signal,
        sample_interval=0.02,
        time_unit="s",
        observable="x",
        rosenstein_emb_dim=8,
        rosenstein_lag=2,
        rosenstein_min_tsep=12,
        rosenstein_min_neighbors=21,
        rosenstein_trajectory_len=30,
        rosenstein_fit="poly",
        rosenstein_fit_offset=3,
        eckmann_emb_dim=9,
        eckmann_matrix_dim=3,
        eckmann_min_neighbors=7,
        eckmann_min_tsep=11,
        random_seed=123,
    )

    assert isinstance(result, TimeSeriesLyapunovResult)
    assert result.largest_exponent == 0.4
    assert result.spectrum == (0.4, 0.0, -2.0)
    assert result.kaplan_yorke_dimension == pytest.approx(2.2)
    assert result.sample_interval == 0.02
    assert result.sample_rate == 50.0
    assert result.exponent_unit == "s^-1"
    assert result.observable == "x"
    assert result.backend_version == "test"
    assert result.largest_sign_agrees_with_spectrum is True
    assert calls["rosenstein"]["tau"] == 0.02
    assert calls["eckmann"]["tau"] == 0.02
    assert calls["rosenstein"]["min_neighbors"] == 21
    assert calls["rosenstein"]["fit_offset"] == 3
    assert calls["eckmann"]["matrix_dim"] == 3
    assert result.to_dict()["spectrum"] == [0.4, 0.0, -2.0]
    assert result.rosenstein_fit_r2 == pytest.approx(1.0)
    assert result.rosenstein_divergence_index_unit == "retained_sample_offset"
    assert result.rosenstein_divergence_time_trajectory[1][0] == pytest.approx(
        0.02
    )


def test_estimator_preserves_numpy_random_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_nolds = SimpleNamespace(
        lyap_r=lambda signal, **kwargs: (
            0.1,
            (
                np.arange(3),
                np.array([0.0, 0.1, 0.2]),
                np.array([0.1, 0.0]),
            ),
        ),
        lyap_e=lambda signal, **kwargs: np.array([0.1, 0.0, -1.0]),
        lyap_r_len=lambda **kwargs: 1,
        lyap_e_len=lambda **kwargs: 1,
    )
    monkeypatch.setattr(module, "require_external", lambda name: fake_nolds)

    np.random.seed(17)
    state_before = np.random.get_state()
    estimate_time_series_lyapunov(
        np.sin(np.linspace(0.0, 20.0, 256)),
        sample_interval=0.1,
        random_seed=99,
    )
    state_after = np.random.get_state()

    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_interval": 0.0}, "sample_interval"),
        (
            {"sample_interval": 1.0, "eckmann_emb_dim": 10},
            "divisible",
        ),
        (
            {"sample_interval": 1.0, "rosenstein_fit": "bad"},
            "rosenstein_fit",
        ),
    ],
)
def test_estimator_rejects_invalid_contract(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_time_series_lyapunov(
            np.sin(np.linspace(0.0, 20.0, 256)),
            **kwargs,
        )


def test_time_series_symbols_are_experimental() -> None:
    assert get_tier(TimeSeriesLyapunovResult) == EXPERIMENTAL
    assert get_tier(estimate_time_series_lyapunov) == EXPERIMENTAL
    assert get_tier(kaplan_yorke_dimension) == EXPERIMENTAL


def test_estimator_accepts_sample_rate_and_guards_pairwise_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_nolds = SimpleNamespace(
        lyap_r=lambda signal, **kwargs: pytest.fail("memory guard was bypassed"),
        lyap_e=lambda signal, **kwargs: pytest.fail("memory guard was bypassed"),
        lyap_r_len=lambda **kwargs: 1,
        lyap_e_len=lambda **kwargs: 1,
    )
    monkeypatch.setattr(module, "require_external", lambda name: fake_nolds)

    with pytest.raises(MemoryError, match="pairwise-distance"):
        estimate_time_series_lyapunov(
            np.sin(np.linspace(0.0, 100.0, 4096)),
            sample_rate=20.0,
            rosenstein_lag=1,
            rosenstein_min_tsep=10,
            max_pairwise_matrix_bytes=1024,
        )


def test_real_nolds_rosenstein_logistic_map() -> None:
    pytest.importorskip("nolds")
    value = 0.123456789
    samples: list[float] = []
    for _ in range(3500):
        value = 4.0 * value * (1.0 - value)
        samples.append(value)

    result = estimate_time_series_lyapunov(
        samples[1000:],
        sample_interval=1.0,
        rosenstein_emb_dim=2,
        rosenstein_lag=1,
        rosenstein_min_tsep=10,
        rosenstein_trajectory_len=8,
        rosenstein_fit="poly",
        rosenstein_fit_offset=0,
        eckmann_emb_dim=3,
        eckmann_matrix_dim=2,
        eckmann_min_neighbors=4,
        eckmann_min_tsep=10,
    )

    assert result.largest_exponent == pytest.approx(np.log(2.0), abs=0.08)
