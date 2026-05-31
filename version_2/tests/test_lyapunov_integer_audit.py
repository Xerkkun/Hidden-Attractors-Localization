"""Tests for F0 audit: integer_qr_benettin Lyapunov method.

F0 audit tests
==============
Verifies:
A. Metadata in LyapunovResult (method_id, derivative_model, q, etc.).
B. Wrapper rejects fractional q with ValueError.
C. Exponents converge to analytic values for a stable linear diagonal system.
D. finite_difference_jacobian still works (no regression).
E. Method registry contains correct implemented/validated flags.
F. Docstring contains the required methodological warning phrase.
G. Documentation does not contain hidden_verified or chaos_verified claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.analysis.lyapunov import (
    LyapunovResult,
    finite_difference_jacobian,
    integer_lyapunov_exponents,
    integer_qr_benettin_lyapunov_exponents,
)
from hidden_attractors.analysis.lyapunov_methods import LYAPUNOV_METHODS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_linear_rhs(x: np.ndarray) -> np.ndarray:
    """x' = diag([-1, -2]) x  →  analytic LEs = [-1, -2]."""
    return np.array([-x[0], -2.0 * x[1]])


def _stable_linear_jacobian(x: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return np.diag([-1.0, -2.0])


# ---------------------------------------------------------------------------
# A. test_integer_qr_benettin_metadata
# ---------------------------------------------------------------------------

class TestIntegerQrBenettinMetadata:
    """A: LyapunovResult carries correct F0 metadata fields."""

    def test_method_id(self) -> None:
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            _stable_linear_jacobian,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=20.0,
        )
        assert result.method_id == "integer_qr_benettin"

    def test_derivative_model(self) -> None:
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            _stable_linear_jacobian,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=20.0,
        )
        assert result.derivative_model == "integer"

    def test_q_field(self) -> None:
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            _stable_linear_jacobian,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=20.0,
        )
        assert result.q == 1.0

    def test_finite_time_local(self) -> None:
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            _stable_linear_jacobian,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=20.0,
        )
        assert result.finite_time_local is True

    def test_orthonormalization(self) -> None:
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            _stable_linear_jacobian,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=20.0,
        )
        assert result.orthonormalization == "qr"

    def test_status_ok(self) -> None:
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            _stable_linear_jacobian,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=20.0,
        )
        assert result.status == "ok"

    def test_jacobian_required(self) -> None:
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            _stable_linear_jacobian,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=20.0,
        )
        assert result.jacobian_required is True

    def test_reference_ids_non_empty(self) -> None:
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            _stable_linear_jacobian,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=20.0,
        )
        assert len(result.reference_ids) > 0

    def test_methodological_warnings_non_empty(self) -> None:
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            _stable_linear_jacobian,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=20.0,
        )
        assert len(result.methodological_warnings) > 0

    def test_lyapunov_result_is_dataclass(self) -> None:
        """LyapunovResult is still a dataclass (backward compat)."""
        import dataclasses
        assert dataclasses.is_dataclass(LyapunovResult)

    def test_integer_lyapunov_exponents_default_metadata(self) -> None:
        """integer_lyapunov_exponents also returns correct metadata."""
        result = integer_lyapunov_exponents(
            _stable_linear_rhs,
            _stable_linear_jacobian,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=20.0,
        )
        assert result.method_id == "integer_qr_benettin"
        assert result.derivative_model == "integer"
        assert result.q == 1.0


# ---------------------------------------------------------------------------
# B. test_integer_qr_benettin_rejects_fractional_q
# ---------------------------------------------------------------------------

class TestIntegerQrBenettinRejectsFractionalQ:
    """B: Wrapper raises ValueError when q != 1."""

    @pytest.mark.parametrize("bad_q", [0.9, 0.99, 0.5, 0.999, 0.95])
    def test_wrapper_raises_for_fractional_q(self, bad_q: float) -> None:
        with pytest.raises(ValueError, match="integer_qr_benettin is valid only for q=1"):
            integer_qr_benettin_lyapunov_exponents(
                _stable_linear_rhs,
                None,
                np.array([1.0, 1.0]),
                h=0.01,
                t_final=5.0,
                q=bad_q,
            )

    def test_base_function_raises_for_fractional_q(self) -> None:
        """integer_lyapunov_exponents also rejects q != 1."""
        with pytest.raises(ValueError, match="integer_qr_benettin is valid only for q=1"):
            integer_lyapunov_exponents(
                _stable_linear_rhs,
                None,
                np.array([1.0, 1.0]),
                h=0.01,
                t_final=5.0,
                q=0.9,
            )

    def test_fractional_method_not_implemented(self) -> None:
        """fractional_cloned_dynamics_abm is still unimplemented in F2."""
        from hidden_attractors.analysis import LyapunovComputationRequest, validate_lyapunov_method_request
        request = LyapunovComputationRequest(
            system=None,
            rhs=_stable_linear_rhs,
            jacobian=None,
            x0=np.array([1.0, 1.0]),
            q=0.99,
            method="fractional_cloned_dynamics_abm",
            h=0.01,
            t_final=5.0,
            memory_mode="full",
        )
        ok, status, _ = validate_lyapunov_method_request(request)
        assert ok is False
        assert status == "method_not_implemented"

    def test_q_exactly_one_does_not_raise(self) -> None:
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            None,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=5.0,
            q=1.0,
        )
        assert result.status == "ok"

    def test_q_within_tolerance_does_not_raise(self) -> None:
        """q = 1.0 + 1e-12 should not raise (within 1e-9 tolerance)."""
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            None,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=5.0,
            q=1.0 + 1e-12,
        )
        assert result.status == "ok"


# ---------------------------------------------------------------------------
# C. test_integer_linear_diagonal_exponents
# ---------------------------------------------------------------------------

class TestIntegerLinearDiagonalExponents:
    """C: Exponents converge to analytic values [-1, -2] for linear stable system."""

    def test_with_analytic_jacobian(self) -> None:
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            _stable_linear_jacobian,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=200.0,
            reorthonormalize_every=10,
        )
        assert result.status == "ok"
        exps = np.sort(result.exponents)[::-1]  # descending
        assert abs(exps[0] - (-1.0)) < 0.1, f"λ₁={exps[0]:.4f}, expected ≈ -1.0"
        assert abs(exps[1] - (-2.0)) < 0.2, f"λ₂={exps[1]:.4f}, expected ≈ -2.0"

    def test_with_finite_difference_jacobian(self) -> None:
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            None,  # use finite differences
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=200.0,
            reorthonormalize_every=10,
        )
        assert result.status == "ok"
        exps = np.sort(result.exponents)[::-1]
        assert abs(exps[0] - (-1.0)) < 0.1
        assert abs(exps[1] - (-2.0)) < 0.2

    def test_exponents_shape(self) -> None:
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            _stable_linear_jacobian,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=50.0,
        )
        assert result.exponents.shape == (2,)

    def test_convergence_history_non_empty(self) -> None:
        result = integer_qr_benettin_lyapunov_exponents(
            _stable_linear_rhs,
            _stable_linear_jacobian,
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=50.0,
        )
        assert result.convergence.shape[0] > 0
        assert result.convergence.shape[1] == 2


# ---------------------------------------------------------------------------
# D. test_finite_difference_jacobian_still_works
# ---------------------------------------------------------------------------

class TestFiniteDifferenceJacobianStillWorks:
    """D: finite_difference_jacobian regression — no existing functionality broken."""

    def test_diagonal_system(self) -> None:
        rhs = lambda x: np.array([-x[0], 2.0 * x[1]])
        J = finite_difference_jacobian(rhs, np.array([1.0, 1.0]))
        assert J.shape == (2, 2)
        assert abs(J[0, 0] - (-1.0)) < 1e-4
        assert abs(J[1, 1] - 2.0) < 1e-4

    def test_shape_preserved(self) -> None:
        rhs = lambda x: np.array([-x[0], x[1], -0.5 * x[2]])
        J = finite_difference_jacobian(rhs, np.ones(3))
        assert J.shape == (3, 3)

    def test_off_diagonal_coupling(self) -> None:
        # x' = [x[1], -x[0]]  → J = [[0, 1], [-1, 0]]
        rhs = lambda x: np.array([x[1], -x[0]])
        J = finite_difference_jacobian(rhs, np.array([0.5, 0.5]))
        assert abs(J[0, 1] - 1.0) < 1e-4
        assert abs(J[1, 0] - (-1.0)) < 1e-4


# ---------------------------------------------------------------------------
# E. test_method_registry_contains_integer_only_validated
# ---------------------------------------------------------------------------

class TestMethodRegistry:
    """E: lyapunov_methods.py registry has correct implemented/validated flags."""

    def test_integer_qr_benettin_implemented(self) -> None:
        info = LYAPUNOV_METHODS["integer_qr_benettin"]
        assert info.implemented is True

    def test_integer_qr_benettin_validated(self) -> None:
        info = LYAPUNOV_METHODS["integer_qr_benettin"]
        assert info.validated is True

    def test_integer_qr_benettin_q_support(self) -> None:
        info = LYAPUNOV_METHODS["integer_qr_benettin"]
        assert "q=1" in info.q_support

    def test_fractional_variational_abm_qr_implemented_in_f2(self) -> None:
        """F2: fractional_variational_abm_qr is now implemented (F2 update)."""
        info = LYAPUNOV_METHODS["fractional_variational_abm_qr"]
        assert info.implemented is True

    def test_fractional_cloned_dynamics_not_implemented(self) -> None:
        info = LYAPUNOV_METHODS["fractional_cloned_dynamics_abm"]
        assert info.implemented is False

    def test_fractional_cloned_dynamics_not_validated(self) -> None:
        info = LYAPUNOV_METHODS["fractional_cloned_dynamics_abm"]
        assert info.validated is False

    def test_registry_has_four_methods(self) -> None:
        assert len(LYAPUNOV_METHODS) == 4

    def test_integer_method_derivative_model(self) -> None:
        info = LYAPUNOV_METHODS["integer_qr_benettin"]
        assert info.derivative_model == "integer"

    def test_fractional_methods_derivative_model(self) -> None:
        for mid in (
            "fractional_variational_abm_qr",
            "fractional_variational_dk2018_block_restart_abm_gs",
            "fractional_cloned_dynamics_abm",
        ):
            assert LYAPUNOV_METHODS[mid].derivative_model == "caputo"


# ---------------------------------------------------------------------------
# F. test_no_fractional_claim_in_integer_doc
# ---------------------------------------------------------------------------

class TestDocstringWarnings:
    """F: Docstring contains the required methodological warning phrase."""

    def test_integer_lyapunov_exponents_docstring_warning(self) -> None:
        doc = integer_lyapunov_exponents.__doc__ or ""
        assert "not a validated Caputo fractional Lyapunov method" in doc

    def test_integer_qr_benettin_wrapper_docstring_warning(self) -> None:
        doc = integer_qr_benettin_lyapunov_exponents.__doc__ or ""
        assert "not a validated Caputo fractional Lyapunov method" in doc

    def test_module_docstring_warning(self) -> None:
        import hidden_attractors.analysis.lyapunov as lmod
        doc = lmod.__doc__ or ""
        assert "NOT a validated Caputo fractional Lyapunov method" in doc

    def test_old_approximate_phrase_gone(self) -> None:
        """'estimates are approximate' was an ambiguous phrase — must be removed."""
        doc = integer_lyapunov_exponents.__doc__ or ""
        assert "estimates are approximate" not in doc

    def test_docstring_mentions_restricted_to_q1(self) -> None:
        doc = integer_lyapunov_exponents.__doc__ or ""
        assert "q = 1" in doc or "q=1" in doc


# ---------------------------------------------------------------------------
# G. test_no_chaos_or_hidden_verified_claims
# ---------------------------------------------------------------------------

class TestNoForbiddenClaims:
    """G: F0 documentation does not assert hidden_verified or chaos_verified."""

    def _load_readme(self) -> str:
        import pathlib
        path = pathlib.Path(__file__).parent.parent / "validation" / "chaos_validation" / "README.md"
        return path.read_text(encoding="utf-8")

    def _load_lyapunov_methods_doc(self) -> str:
        import pathlib
        path = pathlib.Path(__file__).parent.parent / "docs" / "lyapunov_methods.md"
        return path.read_text(encoding="utf-8")

    def test_readme_no_hidden_verified_positive(self) -> None:
        content = self._load_readme()
        # Must not contain 'hidden_verified: true' or 'hidden_verified = True'
        assert "hidden_verified: true" not in content.lower()
        assert "hidden_verified: True" not in content

    def test_readme_no_chaos_verified_positive(self) -> None:
        content = self._load_readme()
        assert "chaos_verified: true" not in content.lower()
        assert "chaos_verified: True" not in content

    def test_readme_has_false_certification(self) -> None:
        """README must contain explicit 'false' claims, not positive assertions."""
        content = self._load_readme()
        assert "false" in content.lower()

    def test_lyapunov_doc_no_hidden_verified_positive(self) -> None:
        content = self._load_lyapunov_methods_doc()
        assert "hidden_verified: true" not in content.lower()

    def test_lyapunov_doc_no_chaos_verified_positive(self) -> None:
        content = self._load_lyapunov_methods_doc()
        assert "chaos_verified: true" not in content.lower()

    def test_lyapunov_doc_has_false_certification(self) -> None:
        content = self._load_lyapunov_methods_doc()
        assert "false" in content.lower()

    def test_lyapunov_result_has_no_verified_flags(self) -> None:
        """LyapunovResult dataclass must not have hidden_verified or chaos_verified fields."""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(LyapunovResult)}
        assert "hidden_verified" not in field_names
        assert "chaos_verified" not in field_names
        assert "fractional_lyapunov_validated" not in field_names
        assert "caputo_lyapunov_validated" not in field_names


# ---------------------------------------------------------------------------
# H. F0 closure — public API exports from analysis package
# ---------------------------------------------------------------------------

class TestF0ClosureExports:
    """H: A1 — analysis package exports F0 symbols correctly."""

    def test_analysis_init_exports_integer_qr_benettin(self) -> None:
        from hidden_attractors.analysis import integer_qr_benettin_lyapunov_exponents
        assert callable(integer_qr_benettin_lyapunov_exponents)

    def test_analysis_init_exports_lyapunov_method_info(self) -> None:
        from hidden_attractors.analysis import LyapunovMethodInfo
        import dataclasses
        assert dataclasses.is_dataclass(LyapunovMethodInfo)

    def test_analysis_init_exports_lyapunov_methods_registry(self) -> None:
        from hidden_attractors.analysis import LYAPUNOV_METHODS
        assert isinstance(LYAPUNOV_METHODS, dict)
        assert LYAPUNOV_METHODS["integer_qr_benettin"].implemented is True

    def test_integer_qr_benettin_in_all(self) -> None:
        import hidden_attractors.analysis as ha
        assert "integer_qr_benettin_lyapunov_exponents" in ha.__all__

    def test_lyapunov_method_info_in_all(self) -> None:
        import hidden_attractors.analysis as ha
        assert "LyapunovMethodInfo" in ha.__all__

    def test_lyapunov_methods_in_all(self) -> None:
        import hidden_attractors.analysis as ha
        assert "LYAPUNOV_METHODS" in ha.__all__


# ---------------------------------------------------------------------------
# I. F0 closure — integer_system_lyapunov_exponents rejects fractional systems
# ---------------------------------------------------------------------------

class TestF0ClosureSystemLyapunov:
    """I: A2 — integer_system_lyapunov_exponents rejects q<1 system objects."""

    # --- Dummy systems ---

    class _DummyFractionalSystem:
        """Minimal system with q=0.99 — must be rejected."""
        q: float = 0.99
        jacobian = None

        def evaluate(self, state: np.ndarray) -> np.ndarray:
            return np.array([-state[0], -2.0 * state[1]])

        def jacobian_matrix(self, state: np.ndarray) -> np.ndarray:
            return np.diag([-1.0, -2.0])

    class _DummyQ1System:
        """Minimal system with q=1.0 — must be allowed."""
        q: float = 1.0
        jacobian = None

        def evaluate(self, state: np.ndarray) -> np.ndarray:
            return np.array([-state[0], -2.0 * state[1]])

        def jacobian_matrix(self, state: np.ndarray) -> np.ndarray:
            return np.diag([-1.0, -2.0])

    class _DummyNoOrderSystem:
        """Minimal system with NO order attribute — backward compat must run."""
        jacobian = None

        def evaluate(self, state: np.ndarray) -> np.ndarray:
            return np.array([-state[0], -2.0 * state[1]])

        def jacobian_matrix(self, state: np.ndarray) -> np.ndarray:
            return np.diag([-1.0, -2.0])

    class _DummyMetadataFractionalSystem:
        """System with q in metadata dict — must be rejected."""
        jacobian = None
        metadata: dict = {"q": 0.95}

        def evaluate(self, state: np.ndarray) -> np.ndarray:
            return np.array([-state[0], -2.0 * state[1]])

        def jacobian_matrix(self, state: np.ndarray) -> np.ndarray:
            return np.diag([-1.0, -2.0])

    def test_integer_system_lyapunov_rejects_fractional_system_object(self) -> None:
        from hidden_attractors.analysis.lyapunov import integer_system_lyapunov_exponents
        with pytest.raises(ValueError, match="valid only for q=1"):
            integer_system_lyapunov_exponents(
                self._DummyFractionalSystem(),
                np.array([1.0, 1.0]),
                h=0.01,
                t_final=5.0,
            )

    def test_integer_system_lyapunov_allows_q1_system(self) -> None:
        from hidden_attractors.analysis.lyapunov import integer_system_lyapunov_exponents
        result = integer_system_lyapunov_exponents(
            self._DummyQ1System(),
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=10.0,
        )
        assert result.status == "ok"

    def test_integer_system_lyapunov_allows_unknown_order_for_backward_compat(self) -> None:
        from hidden_attractors.analysis.lyapunov import integer_system_lyapunov_exponents
        result = integer_system_lyapunov_exponents(
            self._DummyNoOrderSystem(),
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=10.0,
        )
        assert result.status == "ok"

    def test_integer_system_lyapunov_rejects_metadata_fractional_system(self) -> None:
        from hidden_attractors.analysis.lyapunov import integer_system_lyapunov_exponents
        with pytest.raises(ValueError, match="valid only for q=1"):
            integer_system_lyapunov_exponents(
                self._DummyMetadataFractionalSystem(),
                np.array([1.0, 1.0]),
                h=0.01,
                t_final=5.0,
            )


# ---------------------------------------------------------------------------
# J. F1 closure — defensive evaluate/jacobian access
# ---------------------------------------------------------------------------

class TestF1ClosureDefensiveAccess:
    """J: A1 (F1) — integer_system_lyapunov_exponents handles systems without 'jacobian' attr."""

    class _DummyIntegerSystemNoJacobianAttr:
        """System with q=1 and evaluate + jacobian_matrix but NO 'jacobian' attribute."""
        q: float = 1.0

        def evaluate(self, x: np.ndarray) -> np.ndarray:
            return np.array([-x[0], -2.0 * x[1]])

        def jacobian_matrix(self, x: np.ndarray) -> np.ndarray:
            return np.diag([-1.0, -2.0])

    class _DummyIntegerSystemNoEvaluate:
        """System with q=1 but no evaluate — must raise ValueError."""
        q: float = 1.0
        jacobian = None

    def test_allows_object_without_jacobian_attr(self) -> None:
        """System without 'jacobian' attr must run (falls back to finite-diff)."""
        from hidden_attractors.analysis.lyapunov import integer_system_lyapunov_exponents
        result = integer_system_lyapunov_exponents(
            self._DummyIntegerSystemNoJacobianAttr(),
            np.array([1.0, 1.0]),
            h=0.01,
            t_final=10.0,
        )
        assert result.status == "ok"

    def test_raises_if_no_evaluate(self) -> None:
        """System without evaluate() must raise ValueError."""
        from hidden_attractors.analysis.lyapunov import integer_system_lyapunov_exponents
        with pytest.raises(ValueError, match="evaluate"):
            integer_system_lyapunov_exponents(
                self._DummyIntegerSystemNoEvaluate(),
                np.array([1.0, 1.0]),
                h=0.01,
                t_final=5.0,
            )
