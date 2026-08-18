"""Tests for the common Lyapunov API (lyapunov_api.py).

Common API tests
================
Verifies:
1. compute_lyapunov_spectrum works end-to-end for integer_qr_benettin via rhs.
2. Fractional q rejected for integer method.
3. Unknown method raises ValueError with 'unknown_method'.
4. memory_mode='full' rejected for integer method.
5. reorthonormalization_time converts to steps correctly.
6. Both reorthonormalization_time and every → uses every + warning.
7. Public package exports the common API.
8. No hidden_verified / chaos_verified fields in API dataclasses.
9. validate_lyapunov_method_request returns warning for missing analytic Jacobian.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hidden_attractors.analysis.lyapunov_api import (
    LyapunovComputationRequest,
    LyapunovComputationSummary,
    compute_lyapunov_spectrum,
    validate_lyapunov_method_request,
)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rhs(x: np.ndarray) -> np.ndarray:
    """x' = diag([-1, -2]) x  →  analytic LEs = [-1, -2]."""
    return np.array([-x[0], -2.0 * x[1]])


def _jacobian(x: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return np.diag([-1.0, -2.0])


def _make_request(**kwargs: object) -> LyapunovComputationRequest:
    """Build a default-valid LyapunovComputationRequest with overrides."""
    defaults: dict[str, object] = {
        "system": None,
        "rhs": _rhs,
        "jacobian": _jacobian,
        "x0": np.array([1.0, 1.0]),
        "q": 1.0,
        "method": "integer_qr_benettin",
        "h": 0.01,
        "t_final": 10.0,
        "t_burn": 0.0,
        "reorthonormalization_time": None,
        "reorthonormalize_every": None,
        "jacobian_eps": 1e-6,
        "div_threshold": None,
        "memory_mode": "not_applicable",
        "memory_window": None,
        "extra": {},
    }
    defaults.update(kwargs)
    return LyapunovComputationRequest(**defaults)  # type: ignore[arg-type]


def test_system_order_property_failure_is_reported_with_context() -> None:
    class BrokenOrderSystem:
        @property
        def q(self) -> float:
            raise RuntimeError("broken order property")

    request = _make_request(
        system=BrokenOrderSystem(),
        rhs=None,
        q=0.8,
        method="fractional_variational_abm_qr",
        memory_mode="full",
    )
    ok, status, messages = validate_lyapunov_method_request(request)
    assert ok is False
    assert status == "invalid_parameter"
    assert messages == ("system.q could not be read: RuntimeError: broken order property",)


# ---------------------------------------------------------------------------
# 1. test_compute_lyapunov_spectrum_integer_rhs
# ---------------------------------------------------------------------------

class TestComputeSpectrumIntegerRhs:
    """1: End-to-end compute_lyapunov_spectrum via rhs path."""

    def test_method_id_in_result(self) -> None:
        summary = compute_lyapunov_spectrum(
            rhs=_rhs,
            jacobian=_jacobian,
            x0=np.array([1.0, 1.0]),
            q=1.0,
            method="integer_qr_benettin",
            h=0.01,
            t_final=50.0,
        )
        assert summary.result.method_id == "integer_qr_benettin"

    def test_compatibility_status(self) -> None:
        summary = compute_lyapunov_spectrum(
            rhs=_rhs,
            jacobian=_jacobian,
            x0=np.array([1.0, 1.0]),
            q=1.0,
            method="integer_qr_benettin",
            h=0.01,
            t_final=50.0,
        )
        assert summary.compatibility_status == "compatible"

    def test_method_info_method_id(self) -> None:
        summary = compute_lyapunov_spectrum(
            rhs=_rhs,
            jacobian=_jacobian,
            x0=np.array([1.0, 1.0]),
            q=1.0,
            method="integer_qr_benettin",
            h=0.01,
            t_final=50.0,
        )
        assert summary.method_info.method_id == "integer_qr_benettin"

    def test_result_status_ok(self) -> None:
        summary = compute_lyapunov_spectrum(
            rhs=_rhs,
            jacobian=_jacobian,
            x0=np.array([1.0, 1.0]),
            q=1.0,
            method="integer_qr_benettin",
            h=0.01,
            t_final=50.0,
        )
        assert summary.result.status == "ok"

    def test_returns_lyapunov_computation_summary(self) -> None:
        summary = compute_lyapunov_spectrum(
            rhs=_rhs,
            jacobian=_jacobian,
            x0=np.array([1.0, 1.0]),
            q=1.0,
            method="integer_qr_benettin",
            h=0.01,
            t_final=50.0,
        )
        assert isinstance(summary, LyapunovComputationSummary)

    def test_request_summary_has_method(self) -> None:
        summary = compute_lyapunov_spectrum(
            rhs=_rhs,
            jacobian=_jacobian,
            x0=np.array([1.0, 1.0]),
            q=1.0,
            method="integer_qr_benettin",
            h=0.01,
            t_final=50.0,
        )
        assert summary.request_summary["method"] == "integer_qr_benettin"
        assert summary.request_summary["q"] == 1.0
        assert summary.request_summary["h"] == 0.01
        assert summary.request_summary["t_final"] == 50.0


# ---------------------------------------------------------------------------
# 2. test_compute_lyapunov_spectrum_rejects_fractional_q_for_integer_method
# ---------------------------------------------------------------------------

class TestComputeSpectrumRejectsFractionalQ:
    """2: q<1 with integer_qr_benettin raises ValueError."""

    @pytest.mark.parametrize("bad_q", [0.9, 0.99, 0.5])
    def test_raises_value_error_for_bad_q(self, bad_q: float) -> None:
        with pytest.raises(ValueError, match="method_not_valid_for_fractional_caputo"):
            compute_lyapunov_spectrum(
                rhs=_rhs,
                x0=np.array([1.0, 1.0]),
                q=bad_q,
                method="integer_qr_benettin",
                h=0.01,
                t_final=5.0,
            )

    def test_error_mentions_caputo(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            compute_lyapunov_spectrum(
                rhs=_rhs,
                x0=np.array([1.0, 1.0]),
                q=0.95,
                method="integer_qr_benettin",
                h=0.01,
                t_final=5.0,
            )
        assert "caputo" in str(exc_info.value).lower() or "q=1" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. test_unknown_method_rejected
# ---------------------------------------------------------------------------

class TestUnknownMethodRejected:
    """3: Unknown method name raises ValueError with 'unknown_method'."""

    def test_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown_method"):
            compute_lyapunov_spectrum(
                rhs=_rhs,
                x0=np.array([1.0, 1.0]),
                q=1.0,
                method="foo",
                h=0.01,
                t_final=5.0,
            )

    def test_raises_for_misspelled_method(self) -> None:
        with pytest.raises(ValueError, match="unknown_method"):
            compute_lyapunov_spectrum(
                rhs=_rhs,
                x0=np.array([1.0, 1.0]),
                q=1.0,
                method="integer_qr_bennettin",  # deliberate typo
                h=0.01,
                t_final=5.0,
            )


# ---------------------------------------------------------------------------
# 4. test_memory_mode_rejected_for_integer_method
# ---------------------------------------------------------------------------

class TestMemoryModeRejectedForIntegerMethod:
    """5: memory_mode='full' raises ValueError for integer_qr_benettin."""

    @pytest.mark.parametrize("bad_mode", ["full", "window"])
    def test_raises_value_error_for_memory_mode(self, bad_mode: str) -> None:
        with pytest.raises(ValueError, match="memory_mode_not_applicable_for_integer_method"):
            compute_lyapunov_spectrum(
                rhs=_rhs,
                x0=np.array([1.0, 1.0]),
                q=1.0,
                method="integer_qr_benettin",
                h=0.01,
                t_final=5.0,
                memory_mode=bad_mode,
            )

    def test_not_applicable_does_not_raise(self) -> None:
        summary = compute_lyapunov_spectrum(
            rhs=_rhs,
            x0=np.array([1.0, 1.0]),
            q=1.0,
            method="integer_qr_benettin",
            h=0.01,
            t_final=5.0,
            memory_mode="not_applicable",
        )
        assert summary.compatibility_status == "compatible"


# ---------------------------------------------------------------------------
# 6. test_reorthonormalization_time_converts_to_steps
# ---------------------------------------------------------------------------

class TestReorthonormalizationTimeConversion:
    """6: reorthonormalization_time → reorthonormalize_every = round(t/h)."""

    def test_converts_to_10_steps(self) -> None:
        # h=0.01, reorthonormalization_time=0.1 → 10 steps
        summary = compute_lyapunov_spectrum(
            rhs=_rhs,
            jacobian=_jacobian,
            x0=np.array([1.0, 1.0]),
            q=1.0,
            method="integer_qr_benettin",
            h=0.01,
            t_final=5.0,
            reorthonormalization_time=0.1,
        )
        assert summary.request_summary["reorthonormalize_every"] == 10

    def test_converts_to_50_steps(self) -> None:
        # h=0.01, reorthonormalization_time=0.5 → 50 steps
        summary = compute_lyapunov_spectrum(
            rhs=_rhs,
            jacobian=_jacobian,
            x0=np.array([1.0, 1.0]),
            q=1.0,
            method="integer_qr_benettin",
            h=0.01,
            t_final=5.0,
            reorthonormalization_time=0.5,
        )
        assert summary.request_summary["reorthonormalize_every"] == 50

    def test_reorthonormalization_time_stored_in_summary(self) -> None:
        summary = compute_lyapunov_spectrum(
            rhs=_rhs,
            jacobian=_jacobian,
            x0=np.array([1.0, 1.0]),
            q=1.0,
            method="integer_qr_benettin",
            h=0.01,
            t_final=5.0,
            reorthonormalization_time=0.1,
        )
        assert summary.request_summary["reorthonormalization_time"] == 0.1


# ---------------------------------------------------------------------------
# 7. test_both_reorthonormalization_time_and_every_warning
# ---------------------------------------------------------------------------

class TestBothReorthWarning:
    """7: Both reorthonormalization_time and reorthonormalize_every → uses every + warning."""

    def test_uses_reorthonormalize_every(self) -> None:
        summary = compute_lyapunov_spectrum(
            rhs=_rhs,
            jacobian=_jacobian,
            x0=np.array([1.0, 1.0]),
            q=1.0,
            method="integer_qr_benettin",
            h=0.01,
            t_final=5.0,
            reorthonormalization_time=0.1,  # would give 10
            reorthonormalize_every=20,       # must win
        )
        assert summary.request_summary["reorthonormalize_every"] == 20

    def test_warning_added(self) -> None:
        summary = compute_lyapunov_spectrum(
            rhs=_rhs,
            jacobian=_jacobian,
            x0=np.array([1.0, 1.0]),
            q=1.0,
            method="integer_qr_benettin",
            h=0.01,
            t_final=5.0,
            reorthonormalization_time=0.1,
            reorthonormalize_every=20,
        )
        assert any(
            "both_reorthonormalization_time_and_every_provided_using_every" in w
            for w in summary.warnings
        )


# ---------------------------------------------------------------------------
# 8. test_analysis_init_exports_f1_api
# ---------------------------------------------------------------------------

class TestAnalysisInitExportsF1Api:
    """8: Public package exports the F1 API symbols."""

    def test_compute_lyapunov_spectrum_importable(self) -> None:
        from hidden_attractors.analysis import compute_lyapunov_spectrum as cls
        assert callable(cls)

    def test_lyapunov_computation_request_importable(self) -> None:
        from hidden_attractors.analysis import LyapunovComputationRequest as lcr
        assert dataclasses.is_dataclass(lcr)

    def test_lyapunov_computation_summary_importable(self) -> None:
        from hidden_attractors.analysis import LyapunovComputationSummary as lcs
        assert dataclasses.is_dataclass(lcs)

    def test_validate_request_importable(self) -> None:
        from hidden_attractors.analysis import validate_lyapunov_method_request as vrm
        assert callable(vrm)

    def test_f1_symbols_in_all(self) -> None:
        import hidden_attractors.analysis as ha
        for name in (
            "compute_lyapunov_spectrum",
            "LyapunovComputationRequest",
            "LyapunovComputationSummary",
            "validate_lyapunov_method_request",
        ):
            assert name in ha.__all__, f"'{name}' missing from __all__"


# ---------------------------------------------------------------------------
# 9. test_api_no_hidden_or_chaos_verified_fields
# ---------------------------------------------------------------------------

class TestNoForbiddenFieldsInF1:
    """9: F1 dataclasses do not have forbidden verification fields."""

    def test_lyapunov_computation_request_no_verified_fields(self) -> None:
        field_names = {f.name for f in dataclasses.fields(LyapunovComputationRequest)}
        for forbidden in (
            "hidden_verified",
            "chaos_verified",
            "fractional_lyapunov_validated",
            "caputo_lyapunov_validated",
        ):
            assert forbidden not in field_names, f"'{forbidden}' found in LyapunovComputationRequest"

    def test_lyapunov_computation_summary_no_verified_fields(self) -> None:
        field_names = {f.name for f in dataclasses.fields(LyapunovComputationSummary)}
        for forbidden in (
            "hidden_verified",
            "chaos_verified",
            "fractional_lyapunov_validated",
            "caputo_lyapunov_validated",
        ):
            assert forbidden not in field_names, f"'{forbidden}' found in LyapunovComputationSummary"


# ---------------------------------------------------------------------------
# 10. test_validate_request_returns_warning_for_missing_analytic_jacobian
# ---------------------------------------------------------------------------

class TestValidateRequestJacobianWarning:
    """10: validate_lyapunov_method_request warns when jacobian=None and system=None."""

    def test_compatible_with_warning(self) -> None:
        request = _make_request(jacobian=None, system=None)
        ok, status, warnings = validate_lyapunov_method_request(request)
        assert ok is True
        assert status == "compatible"
        assert any(
            "analytic_jacobian_missing_finite_difference_used" in w
            for w in warnings
        )

    def test_no_warning_when_jacobian_provided(self) -> None:
        request = _make_request(jacobian=_jacobian, system=None)
        ok, status, warnings = validate_lyapunov_method_request(request)
        assert ok is True
        assert not any("finite_difference" in w for w in warnings)

    def test_compatible_without_jacobian_still_ok(self) -> None:
        """Finite differences are acceptable — must not be invalid."""
        request = _make_request(jacobian=None, system=None)
        ok, status, _ = validate_lyapunov_method_request(request)
        assert ok is True
        assert status == "compatible"


# ---------------------------------------------------------------------------
# Bonus: validate_lyapunov_method_request edge cases
# ---------------------------------------------------------------------------

class TestValidateRequestEdgeCases:
    """Additional edge-case tests for the validation function."""

    def test_unknown_method_returns_false(self) -> None:
        request = _make_request(method="nonexistent_method")
        ok, status, _ = validate_lyapunov_method_request(request)
        assert ok is False
        assert status == "unknown_method"

    def test_fractional_q_for_integer_method_returns_false(self) -> None:
        request = _make_request(q=0.9)
        ok, status, _ = validate_lyapunov_method_request(request)
        assert ok is False
        assert status == "method_not_valid_for_fractional_caputo"

    def test_memory_full_for_integer_method_returns_false(self) -> None:
        request = _make_request(memory_mode="full")
        ok, status, _ = validate_lyapunov_method_request(request)
        assert ok is False
        assert status == "memory_mode_not_applicable_for_integer_method"

    @pytest.mark.parametrize(
        ("system", "rhs"),
        [
            (None, None),
            (object(), _rhs),
        ],
    )
    def test_requires_exactly_one_vector_field_source(
        self,
        system: object | None,
        rhs: object,
    ) -> None:
        request = _make_request(system=system, rhs=rhs)
        ok, status, warnings = validate_lyapunov_method_request(request)
        assert ok is False
        assert status == "invalid_parameter"
        assert warnings == ("exactly one of system or rhs must be provided.",)

    @pytest.mark.parametrize(
        ("field_name", "bad_value"),
        [
            ("reorthonormalize_every", object()),
            ("reorthonormalization_time", object()),
            ("div_threshold", object()),
        ],
    )
    def test_nonconvertible_optional_returns_invalid_parameter(
        self,
        field_name: str,
        bad_value: object,
    ) -> None:
        request = _make_request(**{field_name: bad_value})
        ok, status, _ = validate_lyapunov_method_request(request)
        assert ok is False
        assert status == "invalid_parameter"

    def test_nonconvertible_memory_window_returns_invalid_parameter(self) -> None:
        request = _make_request(
            q=0.9,
            method="fractional_variational_abm_qr",
            memory_mode="window",
            memory_window=object(),
        )
        ok, status, _ = validate_lyapunov_method_request(request)
        assert ok is False
        assert status == "invalid_parameter"

    @pytest.mark.parametrize("bad_value", [1.5, 0, -1, np.inf, np.nan])
    def test_reorthonormalize_every_requires_positive_integer(
        self,
        bad_value: float,
    ) -> None:
        request = _make_request(reorthonormalize_every=bad_value)
        ok, status, _ = validate_lyapunov_method_request(request)
        assert ok is False
        assert status == "invalid_parameter"

    def test_system_without_analytic_jacobian_warns(self) -> None:
        class SystemWithoutJacobian:
            jacobian = None

            def evaluate(self, state: np.ndarray) -> np.ndarray:
                return _rhs(state)

            def jacobian_matrix(self, state: np.ndarray) -> np.ndarray:
                raise ValueError("No analytic Jacobian is registered.")

        request = _make_request(
            system=SystemWithoutJacobian(),
            rhs=None,
            jacobian=None,
        )
        ok, status, warnings = validate_lyapunov_method_request(request)
        assert ok is True
        assert status == "compatible"
        assert "analytic_jacobian_missing_finite_difference_used" in warnings
