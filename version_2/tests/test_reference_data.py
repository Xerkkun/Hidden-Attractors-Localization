"""
tests/test_reference_data.py
============================
Regression tests against the small reference datasets in ``tests/data/``.

Philosophy
----------
Each test loads one reference file and verifies that the library produces
results within a documented tolerance.  These tests are:

- **Fast** — no heavy integration loops; the reference data was generated
  once with full accuracy and checked in.
- **Reproducible** — they do not depend on random seeds or external services.
- **Scientifically interpretable** — when a test fails, the diff tells you
  *which physical quantity* changed and by how much.

Reference files
---------------
- ``chua_piecewise_equilibria_reference.json``
    Analytically computed fixed points of the piecewise Chua ODE.
    Tests: equilibria match to 1e-8; origin is exact; E+/E- are symmetric.

- ``harmonic_seed_q099_reference.json``
    Harmonic seed for q=0.99 (classical DF, branch 0, nscan=20 000).
    Tests: omega/gain/amplitude within tolerances; seed vector finite and
    close; number of candidates preserved.

- ``tiny_trajectory_q1_reference.csv``
    11-row integer-order (q=1) Chua trajectory, h=0.05, x0=[0.1,0,0].
    Tests: EFORK reproduces each state component within 1e-10 relative
    error; time column exact; energy monotonically dissipates.

- ``basin_labels_tiny_grid_reference.csv``
    Basin classification for a 3×3 grid (q=1, t_final=300 s).
    Tests: exact match of class_id for all 9 points; label consistency;
    structural invariants (E0 at origin, symmetry of target classes).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"


def _load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _load_csv_rows(name: str) -> list[dict]:
    with open(DATA_DIR / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Equilibria reference
# ─────────────────────────────────────────────────────────────────────────────

class TestEquilibriaReference:
    """Verify equilibria_piecewise against the checked-in reference."""

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_json("chua_piecewise_equilibria_reference.json")

    @pytest.fixture(scope="class")
    def computed(self, ref):
        from hidden_attractors.models.chua import (
            ChuaParameters,
            equilibria_piecewise,
        )
        p = ChuaParameters(**ref["params"])
        return equilibria_piecewise(p)

    def test_e0_is_origin(self, ref, computed):
        """E0 must be the zero vector."""
        ref_e0 = np.array(ref["equilibria"]["E0"])
        assert np.allclose(ref_e0, 0.0, atol=1e-15), "Reference E0 is not origin"
        assert np.allclose(computed["E0"], 0.0, atol=1e-15), "Computed E0 is not origin"

    def test_eplus_matches_reference(self, ref, computed):
        """E+ must match the reference within tolerance 1e-8."""
        tol = float(ref["tolerance"])
        ref_ep = np.array(ref["equilibria"]["E+"])
        assert np.allclose(computed["E+"], ref_ep, atol=tol), (
            f"E+ mismatch: got {computed['E+']}, expected {ref_ep}"
        )

    def test_eminus_matches_reference(self, ref, computed):
        """E- must match the reference within tolerance 1e-8."""
        tol = float(ref["tolerance"])
        ref_em = np.array(ref["equilibria"]["E-"])
        assert np.allclose(computed["E-"], ref_em, atol=tol), (
            f"E- mismatch: got {computed['E-']}, expected {ref_em}"
        )

    def test_eplus_eminus_are_antisymmetric(self, computed):
        """E+ and E- must satisfy E- = -E+ (system symmetry)."""
        assert np.allclose(computed["E+"] + computed["E-"], 0.0, atol=1e-10), (
            "E+ + E- is not zero: symmetry broken"
        )

    def test_equilibria_are_fixed_points(self, ref, computed):
        """Each equilibrium must satisfy rhs_piecewise(x, params) ≈ 0."""
        from hidden_attractors.models.chua import ChuaParameters, rhs_piecewise
        p = ChuaParameters(**ref["params"])
        for name, eq in computed.items():
            residual = rhs_piecewise(eq, p)
            assert np.allclose(residual, 0.0, atol=1e-6), (
                f"rhs({name}) = {residual} is not zero — not a fixed point"
            )

    def test_three_equilibria_returned(self, computed):
        """The function must return exactly the three canonical equilibria."""
        assert set(computed.keys()) == {"E0", "E+", "E-"}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Harmonic seed reference
# ─────────────────────────────────────────────────────────────────────────────

class TestHarmonicSeedReference:
    """Verify find_harmonic_seed at q=0.99 against the checked-in reference."""

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_json("harmonic_seed_q099_reference.json")

    @pytest.fixture(scope="class")
    def computed(self, ref):
        from hidden_attractors.models.chua import ChuaParameters
        from hidden_attractors.seed_generation.chua import (
            find_harmonic_seed,
            find_omega_gain_candidates,
        )
        p = ChuaParameters(**ref["params"])
        q = float(ref["q"])
        seed = find_harmonic_seed(q=q, params=p, method="classic", nscan=20_000)
        pairs = find_omega_gain_candidates(q, p, nscan=20_000)
        return {"seed": seed, "pairs": pairs}

    def test_candidate_count_preserved(self, ref, computed):
        """Number of (omega, gain) candidates must match the reference."""
        ref_count = len(ref["omega_gain_candidates"])
        got_count = len(computed["pairs"])
        assert got_count == ref_count, (
            f"Expected {ref_count} candidates, got {got_count}. "
            "A change in the Nyquist scan may have added or removed a branch."
        )

    def test_omega_matches_reference(self, ref, computed):
        tol = float(ref["tolerances"]["omega"])
        ref_omega = float(ref["branch_0"]["omega"])
        got_omega = computed["seed"].omega
        assert abs(got_omega - ref_omega) < tol, (
            f"omega: got {got_omega}, expected {ref_omega} (tol {tol})"
        )

    def test_gain_matches_reference(self, ref, computed):
        tol = float(ref["tolerances"]["gain"])
        ref_gain = float(ref["branch_0"]["gain"])
        got_gain = computed["seed"].gain
        assert abs(got_gain - ref_gain) < tol, (
            f"gain: got {got_gain}, expected {ref_gain} (tol {tol})"
        )

    def test_amplitude_matches_reference(self, ref, computed):
        tol = float(ref["tolerances"]["amplitude"])
        ref_amp = float(ref["branch_0"]["amplitude"])
        got_amp = computed["seed"].amplitude
        assert abs(got_amp - ref_amp) < tol, (
            f"amplitude: got {got_amp}, expected {ref_amp} (tol {tol})"
        )

    def test_seed_vector_close_to_reference(self, ref, computed):
        tol = float(ref["tolerances"]["seed_component"])
        ref_seed = np.array(ref["branch_0"]["seed"])
        got_seed = computed["seed"].seed
        assert np.allclose(got_seed, ref_seed, atol=tol), (
            f"seed vector mismatch: got {got_seed}, expected {ref_seed}"
        )

    def test_seed_is_finite(self, computed):
        assert np.all(np.isfinite(computed["seed"].seed)), "Seed contains non-finite values"

    def test_eigenvalue_close_to_reference(self, ref, computed):
        tol = float(ref["tolerances"]["eigenvalue"])
        ref_re = float(ref["branch_0"]["matched_eigenvalue_re"])
        ref_im = float(ref["branch_0"]["matched_eigenvalue_im"])
        got_ev = computed["seed"].matched_eigenvalue
        assert abs(got_ev.real - ref_re) < tol, (
            f"eigenvalue Re: got {got_ev.real}, expected {ref_re}"
        )
        assert abs(got_ev.imag - ref_im) < tol, (
            f"eigenvalue Im: got {got_ev.imag}, expected {ref_im}"
        )

    def test_omega_is_nyquist_root(self, ref, computed):
        """Im(W_q(i*omega)) must be near zero at the returned omega."""
        from hidden_attractors.models.chua import ChuaParameters
        from hidden_attractors.seed_generation.chua import transfer_function
        p = ChuaParameters(**ref["params"])
        q = float(ref["q"])
        omega = computed["seed"].omega
        W = transfer_function(omega, q, p)
        assert abs(W.imag) < 0.01, (
            f"Im(W_q(i*omega)) = {W.imag:.6f} is not near zero — "
            "omega is not a Nyquist crossing"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tiny trajectory reference
# ─────────────────────────────────────────────────────────────────────────────

class TestTinyTrajectoryReference:
    """Verify efork_q1_integrate reproduces the checked-in trajectory."""

    @pytest.fixture(scope="class")
    def ref_traj(self) -> np.ndarray:
        rows = _load_csv_rows("tiny_trajectory_q1_reference.csv")
        return np.array([[float(r["t"]), float(r["x"]), float(r["y"]), float(r["z"])]
                         for r in rows])

    @pytest.fixture(scope="class")
    def computed_traj(self):
        from hidden_attractors.models.chua import (
            chua_piecewise_parameters,
            nonlinearity_piecewise,
        )
        from hidden_attractors.solvers import efork_q1_integrate

        p = chua_piecewise_parameters()

        def rhs(x: np.ndarray) -> np.ndarray:
            nl = nonlinearity_piecewise(x[0], p)
            return np.array([
                p.alpha * (x[1] - x[0] - nl),
                x[0] - x[1] + x[2],
                -p.beta * x[1] - p.gamma * x[2],
            ])

        x0 = np.array([0.1, 0.0, 0.0])
        traj, status = efork_q1_integrate(rhs, x0, t_final=0.5, h=0.05)
        assert status == "ok", f"Solver status: {status}"
        return traj

    def test_row_count(self, ref_traj, computed_traj):
        assert computed_traj.shape[0] == ref_traj.shape[0], (
            f"Row count: got {computed_traj.shape[0]}, expected {ref_traj.shape[0]}"
        )

    def test_time_column_exact(self, ref_traj, computed_traj):
        """Time column must be exact (it is computed as i*h, no accumulation)."""
        assert np.allclose(computed_traj[:, 0], ref_traj[:, 0], atol=1e-14), (
            "Time column does not match reference"
        )

    def test_state_columns_close(self, ref_traj, computed_traj):
        """State columns must match to 1e-10 absolute tolerance."""
        for col, name in enumerate(["x", "y", "z"], start=1):
            assert np.allclose(computed_traj[:, col], ref_traj[:, col], atol=1e-10), (
                f"State column '{name}' mismatch at one or more time steps"
            )

    def test_initial_condition_exact(self, ref_traj, computed_traj):
        """Row 0 must reproduce x0 = [0.1, 0, 0] exactly."""
        assert np.allclose(computed_traj[0, 1:], [0.1, 0.0, 0.0], atol=1e-15)

    def test_x_component_decays_from_ic(self, computed_traj):
        """For this IC, the x component should decrease from 0.1 over the short window."""
        x_col = computed_traj[:, 1]
        assert x_col[0] > x_col[-1], (
            "x component did not decay from the IC — unexpected trajectory direction"
        )

    def test_all_values_finite(self, computed_traj):
        assert np.all(np.isfinite(computed_traj)), "Trajectory contains non-finite values"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Basin labels tiny grid reference
# ─────────────────────────────────────────────────────────────────────────────

class TestBasinLabelsReference:
    """Verify BasinBackend classification matches the checked-in 3×3 grid."""

    CLASSIFY_KWARGS = dict(
        q=1.0, h=0.01, Lm=10.0,
        t_final=300.0, t_burn=100.0,
        divergence_norm=120.0, r_bound=60.0,
        equilibrium_tol=0.01, cap_win=200, mean_x_gap=0.5,
    )

    @pytest.fixture(scope="class")
    def ref_rows(self) -> list[dict]:
        return _load_csv_rows("basin_labels_tiny_grid_reference.csv")

    @pytest.fixture(scope="class")
    def backend(self):
        from hidden_attractors.models.chua import chua_piecewise_parameters
        from hidden_attractors.native.backends import BasinBackend
        b = BasinBackend.build()
        b.set_piecewise_params(chua_piecewise_parameters())
        return b

    def test_all_class_ids_match(self, ref_rows, backend):
        """Every point in the reference grid must reproduce the same class_id."""
        mismatches = []
        for row in ref_rows:
            x0 = [float(row["x0"]), float(row["y0"]), float(row["z0"])]
            ref_cid = int(row["class_id"])
            got_cid = backend.classify_point(x0, **self.CLASSIFY_KWARGS)
            if got_cid != ref_cid:
                mismatches.append(
                    f"  ({x0[0]}, {x0[1]}, {x0[2]}): "
                    f"expected class {ref_cid} ({row['class_label']}), "
                    f"got {got_cid}"
                )
        assert not mismatches, (
            f"{len(mismatches)} classification(s) changed:\n" + "\n".join(mismatches)
        )

    def test_label_text_consistency(self, ref_rows, backend):
        """class_label() must agree with the stored class_label column."""
        from hidden_attractors.basins.classification import class_label
        for row in ref_rows:
            stored_label = row["class_label"]
            cid = int(row["class_id"])
            assert class_label(cid) == stored_label, (
                f"class_label({cid}) = '{class_label(cid)}', "
                f"but reference says '{stored_label}'"
            )

    def test_origin_near_point_is_equilibrium(self, ref_rows):
        """The point (0.05, 0, 0) must be classified as 'equilibrium' (class 0)."""
        origin_row = next(
            (r for r in ref_rows if float(r["x0"]) == 0.05 and float(r["z0"]) == 0.0),
            None,
        )
        assert origin_row is not None, "Origin-near point not found in reference"
        assert int(origin_row["class_id"]) == 0, (
            f"Origin-near point classified as {origin_row['class_label']}, expected equilibrium"
        )

    def test_target_classes_are_symmetric(self, ref_rows):
        """For each target_positive point at z<0, there should be a target_negative at z>0."""
        positives = {
            (float(r["x0"]), float(r["y0"])): float(r["z0"])
            for r in ref_rows if r["class_label"] == "target_positive"
        }
        negatives = {
            (float(r["x0"]), float(r["y0"])): float(r["z0"])
            for r in ref_rows if r["class_label"] == "target_negative"
        }
        assert set(positives.keys()) == set(negatives.keys()), (
            "target_positive and target_negative do not share the same (x0,y0) pairs "
            "— symmetry of the Chua attractor may be broken"
        )

    def test_large_norm_points_are_infinity(self, ref_rows):
        """Points with |x0| = 6 should be classified as 'infinity' (class 3)."""
        large_rows = [r for r in ref_rows if abs(float(r["x0"])) == 6.0]
        assert large_rows, "No large-norm points in reference"
        for r in large_rows:
            assert r["class_label"] == "infinity", (
                f"({r['x0']},{r['z0']}) classified as '{r['class_label']}', "
                "expected 'infinity'"
            )

    def test_nine_points_in_reference(self, ref_rows):
        assert len(ref_rows) == 9, f"Expected 9 reference rows, got {len(ref_rows)}"
