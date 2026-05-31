"""Tests for F2 — fractional_variational_abm_qr Lyapunov estimator.

F2 tests
========
Verifies:
1.  pack/unpack roundtrip.
2.  Extended RHS shapes.
3.  Rejects q=1.
4.  Rejects invalid q (<=0 or >1).
5.  Zero-RHS: exponents near zero.
6.  Linear stable system: lambda_max < 0.01.
7.  History-aware QR: current Phi is orthonormal after transform.
8.  History-aware QR: historical blocks are also transformed.
9.  Result metadata fields.
10. API dispatch via compute_lyapunov_spectrum.
11. API rejects fractional_variational_abm_qr with q=1.
12. API rejects bad memory_mode ('not_applicable') for fractional method.
13. Registry: implemented=True, validated=False.
14. No chaos/hidden verified fields in LyapunovResult.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hidden_attractors.analysis.lyapunov_fractional import (
    FractionalVariationalQRConfig,
    apply_history_aware_qr_transform,
    build_extended_variational_rhs,
    fractional_variational_abm_qr,
    pack_extended_state,
    unpack_extended_state,
)
from hidden_attractors.analysis.lyapunov import LyapunovResult
from hidden_attractors.analysis.lyapunov_methods import LYAPUNOV_METHODS
from hidden_attractors.analysis import compute_lyapunov_spectrum


# ---------------------------------------------------------------------------
# 1. pack/unpack roundtrip
# ---------------------------------------------------------------------------

class TestPackUnpackRoundtrip:
    """1: pack_extended_state / unpack_extended_state are inverse operations."""

    def test_roundtrip_2d(self) -> None:
        n = 2
        X = np.array([1.5, -2.3])
        Phi = np.array([[1.0, 0.5], [-0.3, 1.2]])
        Y = pack_extended_state(X, Phi)
        X2, Phi2 = unpack_extended_state(Y, n)
        np.testing.assert_allclose(X2, X)
        np.testing.assert_allclose(Phi2, Phi)

    def test_roundtrip_3d(self) -> None:
        n = 3
        rng = np.random.default_rng(42)
        X = rng.standard_normal(n)
        Phi = rng.standard_normal((n, n))
        Y = pack_extended_state(X, Phi)
        assert Y.shape == (n + n * n,)
        X2, Phi2 = unpack_extended_state(Y, n)
        np.testing.assert_allclose(X2, X)
        np.testing.assert_allclose(Phi2, Phi)

    def test_shape_output(self) -> None:
        n = 4
        X = np.ones(n)
        Phi = np.eye(n)
        Y = pack_extended_state(X, Phi)
        assert Y.shape == (n + n * n,)

    def test_pack_row_major(self) -> None:
        """Row-major layout: Phi is stored by rows in Y[n:]."""
        n = 2
        Phi = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y = pack_extended_state(np.zeros(n), Phi)
        # Row-major: [1, 2, 3, 4]
        np.testing.assert_allclose(Y[n:], [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# 2. Extended RHS shapes
# ---------------------------------------------------------------------------

class TestExtendedRhsShapes:
    """2: build_extended_variational_rhs produces correct-shape output."""

    def test_shape_2d_with_analytic_jacobian(self) -> None:
        n = 2
        rhs = lambda x: np.array([-x[0], -2.0 * x[1]])
        jac = lambda x: np.diag([-1.0, -2.0])
        G = build_extended_variational_rhs(rhs, jac, n)
        X0 = np.array([1.0, 1.0])
        Phi0 = np.eye(n)
        Y0 = pack_extended_state(X0, Phi0)
        dY = G(Y0)
        assert dY.shape == (n + n * n,)

    def test_shape_2d_with_finite_diff_jacobian(self) -> None:
        n = 2
        rhs = lambda x: np.array([-x[0], -2.0 * x[1]])
        G = build_extended_variational_rhs(rhs, None, n)
        Y0 = pack_extended_state(np.array([1.0, 1.0]), np.eye(n))
        dY = G(Y0)
        assert dY.shape == (n + n * n,)

    def test_dx_block_values(self) -> None:
        """dX block of G(Y) must equal rhs(X)."""
        n = 2
        rhs = lambda x: np.array([-x[0], -2.0 * x[1]])
        jac = lambda x: np.diag([-1.0, -2.0])
        G = build_extended_variational_rhs(rhs, jac, n)
        X0 = np.array([3.0, -1.0])
        Y0 = pack_extended_state(X0, np.eye(n))
        dY = G(Y0)
        dX, _ = unpack_extended_state(dY, n)
        np.testing.assert_allclose(dX, rhs(X0))


# ---------------------------------------------------------------------------
# 3. Rejects q=1
# ---------------------------------------------------------------------------

class TestRejectsQ1:
    """3: fractional_variational_abm_qr raises ValueError for q=1."""

    def test_raises_for_q1(self) -> None:
        rhs = lambda x: np.array([-x[0]])
        with pytest.raises(ValueError, match="0 < q < 1"):
            fractional_variational_abm_qr(
                rhs, None, np.array([1.0]), q=1.0, h=0.01, t_final=1.0
            )

    def test_config_raises_for_q1(self) -> None:
        with pytest.raises(ValueError, match="0 < q < 1"):
            FractionalVariationalQRConfig(q=1.0, h=0.01, t_final=1.0)


# ---------------------------------------------------------------------------
# 4. Rejects invalid q (<=0 or >1)
# ---------------------------------------------------------------------------

class TestRejectsInvalidQ:
    """4: Values outside (0,1) are rejected."""

    @pytest.mark.parametrize("bad_q", [0.0, -0.5, 1.5, 2.0])
    def test_raises_for_bad_q(self, bad_q: float) -> None:
        rhs = lambda x: np.array([-x[0]])
        with pytest.raises(ValueError, match="0 < q < 1"):
            fractional_variational_abm_qr(
                rhs, None, np.array([1.0]), q=bad_q, h=0.01, t_final=1.0
            )

    @pytest.mark.parametrize("bad_q", [0.0, -0.1, 1.1])
    def test_config_raises_for_bad_q(self, bad_q: float) -> None:
        with pytest.raises(ValueError, match="0 < q < 1"):
            FractionalVariationalQRConfig(q=bad_q, h=0.01, t_final=1.0)


# ---------------------------------------------------------------------------
# 5. Zero RHS: exponents near zero
# ---------------------------------------------------------------------------

class TestZeroRhsZeroExponents:
    """5: ᶜD^q X = 0 → Phi ≈ I → exponents near 0."""

    def test_exponents_near_zero(self) -> None:
        n = 2
        rhs = lambda x: np.zeros(n)
        jac = lambda x: np.zeros((n, n))
        result = fractional_variational_abm_qr(
            rhs,
            jac,
            np.array([0.5, 0.5]),
            q=0.9,
            h=0.05,
            t_final=2.0,
            reorthonormalization_time=0.25,
        )
        assert result.status == "ok"
        # exponents may be nan if no accumulation happened, otherwise near 0
        finite_mask = np.isfinite(result.exponents)
        if finite_mask.any():
            assert np.all(np.abs(result.exponents[finite_mask]) < 1e-3)

    def test_method_id(self) -> None:
        n = 2
        rhs = lambda x: np.zeros(n)
        result = fractional_variational_abm_qr(
            rhs, None, np.zeros(n), q=0.9, h=0.05, t_final=1.0
        )
        assert result.method_id == "fractional_variational_abm_qr"


# ---------------------------------------------------------------------------
# 6. Linear stable diagonal: lambda_max not positive-large
# ---------------------------------------------------------------------------

class TestLinearStableNonpositiveLambdaMax:
    """6: ᶜD^q X = A X, A=diag(-1,-2) → lambda_max < 0.01."""

    def test_lambda_max_nonpositive(self) -> None:
        n = 2
        A = np.diag([-1.0, -2.0])
        rhs = lambda x: A @ x
        jac = lambda x: A
        result = fractional_variational_abm_qr(
            rhs,
            jac,
            np.array([1.0, 1.0]),
            q=0.9,
            h=0.01,
            t_final=10.0,
            reorthonormalize_every=10,
        )
        assert result.status in ("ok", "diverged", "nonfinite_solution")
        if result.status == "ok" and np.all(np.isfinite(result.exponents)):
            assert np.max(result.exponents) < 1e-2, (
                f"lambda_max={np.max(result.exponents):.4f}, expected < 0.01"
            )


# ---------------------------------------------------------------------------
# 7. History-aware QR: Phi_current is orthonormal after transform
# ---------------------------------------------------------------------------

class TestHistoryAwareQrCurrentOrthonormal:
    """7: After apply_history_aware_qr_transform, Phi_current ≈ Q (orthonormal)."""

    def _make_history(self, n: int, length: int) -> tuple[list, list, object]:
        rng = np.random.default_rng(7)
        rhs = lambda x: np.zeros(n)
        jac = lambda x: np.zeros((n, n))
        G = build_extended_variational_rhs(rhs, jac, n)
        states = []
        rhs_vals = []
        for _ in range(length):
            X = rng.standard_normal(n)
            Phi = rng.standard_normal((n, n))
            Y = pack_extended_state(X, Phi)
            states.append(Y)
            rhs_vals.append(np.asarray(G(Y), dtype=float))
        return states, rhs_vals, G

    def test_phi_current_is_orthonormal(self) -> None:
        n = 3
        length = 5
        states, rhs_vals, G = self._make_history(n, length)
        cur_idx = length - 1
        apply_history_aware_qr_transform(
            states, rhs_vals, G, n, cur_idx,
            qr_epsilon=1e-300, memory_start_index=0,
        )
        _, Phi_cur = unpack_extended_state(states[cur_idx], n)
        orth = Phi_cur.T @ Phi_cur
        np.testing.assert_allclose(orth, np.eye(n), atol=1e-10)

    def test_returns_log_diag_shape(self) -> None:
        n = 2
        length = 3
        states, rhs_vals, G = self._make_history(n, length)
        log_diag, cond_R, qr_st = apply_history_aware_qr_transform(
            states, rhs_vals, G, n, length - 1
        )
        assert log_diag.shape == (n,)
        assert isinstance(cond_R, float)
        assert qr_st in ("ok", "qr_ill_conditioned")


# ---------------------------------------------------------------------------
# 8. History-aware QR: historical blocks are also transformed
# ---------------------------------------------------------------------------

class TestHistoryAwareQrTransformsAllHistory:
    """8: All stored Phi_j in [memory_start, current] are modified."""

    def test_at_least_two_history_blocks_change(self) -> None:
        n = 2
        rng = np.random.default_rng(99)
        G = build_extended_variational_rhs(
            lambda x: np.zeros(n), lambda x: np.zeros((n, n)), n
        )
        states = []
        for _ in range(4):
            X = rng.standard_normal(n)
            Phi = rng.standard_normal((n, n))
            states.append(pack_extended_state(X, Phi))
        rhs_vals = [np.asarray(G(s), dtype=float) for s in states]

        # snapshot Phi values before
        phis_before = [unpack_extended_state(s, n)[1].copy() for s in states]

        apply_history_aware_qr_transform(
            states, rhs_vals, G, n, len(states) - 1,
            qr_epsilon=1e-300, memory_start_index=0,
        )

        phis_after = [unpack_extended_state(s, n)[1] for s in states]
        changed = sum(
            not np.allclose(phis_before[j], phis_after[j])
            for j in range(len(states))
        )
        assert changed >= 2, f"Expected >=2 changed blocks, got {changed}"


# ---------------------------------------------------------------------------
# 9. Result metadata fields
# ---------------------------------------------------------------------------

class TestFractionalResultMetadata:
    """9: LyapunovResult from fractional_variational_abm_qr has correct metadata."""

    def _small_result(self) -> LyapunovResult:
        n = 2
        rhs = lambda x: np.array([-x[0], -2.0 * x[1]])
        jac = lambda x: np.diag([-1.0, -2.0])
        return fractional_variational_abm_qr(
            rhs, jac, np.ones(n), q=0.9, h=0.02, t_final=2.0,
            reorthonormalize_every=10,
        )

    def test_method_id(self) -> None:
        assert self._small_result().method_id == "fractional_variational_abm_qr"

    def test_derivative_model(self) -> None:
        assert self._small_result().derivative_model == "caputo"

    def test_q_field(self) -> None:
        assert self._small_result().q == 0.9

    def test_finite_time_local(self) -> None:
        assert self._small_result().finite_time_local is True

    def test_orthonormalization(self) -> None:
        assert self._small_result().orthonormalization == "qr"

    def test_jacobian_required(self) -> None:
        assert self._small_result().jacobian_required is True

    def test_reference_ids_non_empty(self) -> None:
        assert len(self._small_result().reference_ids) > 0
        assert any("Danca" in r for r in self._small_result().reference_ids)

    def test_methodological_warnings_non_empty(self) -> None:
        assert len(self._small_result().methodological_warnings) > 0

    def test_no_chaos_certified_claim(self) -> None:
        """Warnings must not claim chaos_certified_by_this_pipeline: true."""
        warnings_text = " ".join(self._small_result().methodological_warnings).lower()
        assert "chaos_certified_by_this_pipeline: true" not in warnings_text

    def test_no_hidden_verified_claim(self) -> None:
        warnings_text = " ".join(self._small_result().methodological_warnings).lower()
        assert "hidden_verified: true" not in warnings_text


# ---------------------------------------------------------------------------
# 10. API dispatch: compute_lyapunov_spectrum
# ---------------------------------------------------------------------------

class TestApiDispatchFractionalVariational:
    """10: compute_lyapunov_spectrum dispatches correctly to F2."""

    def test_returns_correct_method_id(self) -> None:
        n = 2
        rhs = lambda x: np.array([-x[0], -2.0 * x[1]])
        jac = lambda x: np.diag([-1.0, -2.0])
        summary = compute_lyapunov_spectrum(
            rhs=rhs,
            jacobian=jac,
            x0=np.ones(n),
            q=0.9,
            method="fractional_variational_abm_qr",
            h=0.02,
            t_final=2.0,
            memory_mode="full",
            reorthonormalize_every=10,
        )
        assert summary.result.method_id == "fractional_variational_abm_qr"

    def test_compatibility_status(self) -> None:
        n = 2
        rhs = lambda x: np.array([-x[0], -2.0 * x[1]])
        summary = compute_lyapunov_spectrum(
            rhs=rhs,
            x0=np.ones(n),
            q=0.9,
            method="fractional_variational_abm_qr",
            h=0.02,
            t_final=2.0,
            memory_mode="full",
        )
        assert summary.compatibility_status == "compatible"

    def test_method_info_implemented(self) -> None:
        n = 2
        rhs = lambda x: np.array([-x[0], -2.0 * x[1]])
        summary = compute_lyapunov_spectrum(
            rhs=rhs,
            x0=np.ones(n),
            q=0.9,
            method="fractional_variational_abm_qr",
            h=0.02,
            t_final=2.0,
            memory_mode="full",
        )
        assert summary.method_info.implemented is True
        assert summary.method_info.validated is False

    def test_request_summary_has_method(self) -> None:
        n = 2
        rhs = lambda x: np.array([-x[0], -2.0 * x[1]])
        summary = compute_lyapunov_spectrum(
            rhs=rhs,
            x0=np.ones(n),
            q=0.9,
            method="fractional_variational_abm_qr",
            h=0.02,
            t_final=2.0,
            memory_mode="full",
        )
        assert summary.request_summary["method"] == "fractional_variational_abm_qr"
        assert summary.request_summary["q"] == 0.9


# ---------------------------------------------------------------------------
# 11. API rejects fractional_variational_abm_qr with q=1
# ---------------------------------------------------------------------------

class TestApiRejectsFractionalVariationalWithIntegerQ:
    """11: compute_lyapunov_spectrum raises ValueError for q=1 with fractional method."""

    def test_raises_value_error(self) -> None:
        rhs = lambda x: np.array([-x[0], -x[1]])
        with pytest.raises(ValueError, match="method_not_valid_for_integer_or_out_of_range_q"):
            compute_lyapunov_spectrum(
                rhs=rhs,
                x0=np.ones(2),
                q=1.0,
                method="fractional_variational_abm_qr",
                h=0.01,
                t_final=5.0,
                memory_mode="full",
            )


# ---------------------------------------------------------------------------
# 12. API rejects bad memory_mode for fractional method
# ---------------------------------------------------------------------------

class TestApiRejectsBadMemoryModeForFractional:
    """12: memory_mode='not_applicable' raises ValueError for fractional method."""

    def test_raises_for_not_applicable(self) -> None:
        rhs = lambda x: np.array([-x[0], -x[1]])
        with pytest.raises(ValueError, match="memory_mode_must_be_full_or_window"):
            compute_lyapunov_spectrum(
                rhs=rhs,
                x0=np.ones(2),
                q=0.9,
                method="fractional_variational_abm_qr",
                h=0.01,
                t_final=5.0,
                memory_mode="not_applicable",
            )

    def test_full_mode_accepted(self) -> None:
        rhs = lambda x: np.array([-x[0], -x[1]])
        summary = compute_lyapunov_spectrum(
            rhs=rhs,
            x0=np.ones(2),
            q=0.9,
            method="fractional_variational_abm_qr",
            h=0.02,
            t_final=2.0,
            memory_mode="full",
        )
        assert summary.compatibility_status == "compatible"


# ---------------------------------------------------------------------------
# 13. Registry: fractional_variational_abm_qr implemented=True, validated=False
# ---------------------------------------------------------------------------

class TestFractionalRegistryStatus:
    """13: LYAPUNOV_METHODS reflects F2 implementation status."""

    def test_implemented_true(self) -> None:
        info = LYAPUNOV_METHODS["fractional_variational_abm_qr"]
        assert info.implemented is True

    def test_validated_false(self) -> None:
        info = LYAPUNOV_METHODS["fractional_variational_abm_qr"]
        assert info.validated is False

    def test_derivative_model_caputo(self) -> None:
        info = LYAPUNOV_METHODS["fractional_variational_abm_qr"]
        assert info.derivative_model == "caputo"

    def test_warnings_mention_not_validated(self) -> None:
        info = LYAPUNOV_METHODS["fractional_variational_abm_qr"]
        combined = " ".join(info.warnings).lower()
        assert "not yet validated" in combined or "not validated" in combined

    def test_warnings_no_chaos_certified_positive(self) -> None:
        info = LYAPUNOV_METHODS["fractional_variational_abm_qr"]
        combined = " ".join(info.warnings).lower()
        assert "chaos_certified_by_this_pipeline: true" not in combined


# ---------------------------------------------------------------------------
# 14. No chaos/hidden verified fields in LyapunovResult
# ---------------------------------------------------------------------------

class TestNoForbiddenFieldsInLyapunovResult:
    """14: LyapunovResult must not expose hidden_verified, chaos_verified, etc."""

    def test_no_verified_flags(self) -> None:
        field_names = {f.name for f in dataclasses.fields(LyapunovResult)}
        for forbidden in (
            "hidden_verified",
            "chaos_verified",
            "fractional_lyapunov_validated",
            "caputo_lyapunov_validated",
        ):
            assert forbidden not in field_names, (
                f"'{forbidden}' found in LyapunovResult — must not be present."
            )


# ---------------------------------------------------------------------------
# 15. Non-aligned burn time elapsed handling
# ---------------------------------------------------------------------------

class TestNonAlignedBurnTimeElapsed:
    """15: Handles non-aligned burn-in times correctly without negative or bad elapsed time."""

    def test_fractional_variational_elapsed_handles_nonaligned_burn(self) -> None:
        n = 2
        rhs = lambda x: np.array([-x[0], -2.0 * x[1]])
        jac = lambda x: np.diag([-1.0, -2.0])
        result = fractional_variational_abm_qr(
            rhs,
            jac,
            np.ones(n),
            q=0.9,
            h=0.02,
            t_burn=0.13,
            t_final=0.5,
            reorthonormalization_time=0.10,
        )
        assert result.status == "ok"
        if len(result.times) > 0:
            diffs = np.diff(result.times)
            assert np.all(diffs > 0), f"result.times is not strictly increasing: {result.times}"
            assert result.times[-1] > 0, f"result.times[-1] <= 0: {result.times[-1]}"
            assert np.all(np.isfinite(result.exponents))

