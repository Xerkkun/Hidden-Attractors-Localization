"""Tests for evaluate_target_match (centroid_distance, bbox_overlap, nn_percentile)."""

import numpy as np
import pytest
from src.verification.hiddenness import evaluate_target_match


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circle_cloud(n: int, radius: float, center: np.ndarray, seed: int = 0) -> np.ndarray:
    """2-D points uniformly distributed on a circle (useful as a limit-cycle proxy)."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)]) + center
    return pts


# ---------------------------------------------------------------------------
# centroid_distance
# ---------------------------------------------------------------------------

class TestCentroidDistance:
    def test_identical_clouds_match(self):
        ref = _circle_cloud(200, 3.0, np.array([0.0, 0.0]))
        assert evaluate_target_match(ref, ref.copy(), metric="centroid_distance", tolerance=0.1)

    def test_close_centroids_match(self):
        ref = _circle_cloud(200, 3.0, np.array([0.0, 0.0]))
        probe = _circle_cloud(200, 3.0, np.array([0.05, 0.05]))  # centroid shift < tol
        assert evaluate_target_match(probe, ref, metric="centroid_distance", tolerance=0.2)

    def test_far_centroids_no_match(self):
        ref = _circle_cloud(200, 3.0, np.array([0.0, 0.0]))
        probe = _circle_cloud(200, 3.0, np.array([10.0, 0.0]))
        assert not evaluate_target_match(probe, ref, metric="centroid_distance", tolerance=0.5)

    def test_empty_trajectory(self):
        ref = _circle_cloud(50, 1.0, np.array([0.0, 0.0]))
        assert not evaluate_target_match(np.empty((0, 2)), ref, metric="centroid_distance")

    def test_empty_reference(self):
        probe = _circle_cloud(50, 1.0, np.array([0.0, 0.0]))
        assert not evaluate_target_match(probe, np.empty((0, 2)), metric="centroid_distance")


# ---------------------------------------------------------------------------
# bbox_overlap
# ---------------------------------------------------------------------------

class TestBboxOverlap:
    def test_overlapping_clouds_match(self):
        ref = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        probe = np.array([[0.4, 0.4], [0.9, 0.9]])
        assert evaluate_target_match(probe, ref, metric="bbox_overlap")

    def test_disjoint_clouds_no_match(self):
        ref = np.array([[0.0, 0.0], [1.0, 1.0]])
        probe = np.array([[5.0, 5.0], [6.0, 6.0]])
        assert not evaluate_target_match(probe, ref, metric="bbox_overlap")

    def test_touching_boxes_overlap(self):
        # boxes share an edge at x=1 → overlap returns True (touching counts)
        ref = np.array([[0.0, 0.0], [1.0, 1.0]])
        probe = np.array([[1.0, 0.5], [2.0, 1.5]])
        assert evaluate_target_match(probe, ref, metric="bbox_overlap")


# ---------------------------------------------------------------------------
# nn_percentile
# ---------------------------------------------------------------------------

class TestNNPercentile:
    """The nn_percentile metric is the primary target of this test suite."""

    def test_same_cloud_matches(self):
        """A cloud compared against itself must always match."""
        ref = _circle_cloud(300, 2.0, np.array([0.0, 0.0]))
        assert evaluate_target_match(
            ref, ref.copy(), metric="nn_percentile", tolerance=1e-6, nn_percentile=90.0
        )

    def test_nearby_cloud_matches(self):
        """Two clouds on the same orbit with small angular offset must match."""
        ref = _circle_cloud(300, 2.0, np.array([0.0, 0.0]), seed=0)
        probe = _circle_cloud(300, 2.0, np.array([0.0, 0.0]), seed=42)
        # Max NN distance for two uniform samples from the same circle of radius 2
        # should be << 1 with 300 points each
        assert evaluate_target_match(
            probe, ref, metric="nn_percentile", tolerance=0.5, nn_percentile=90.0
        )

    def test_different_attractor_no_match(self):
        """A probe on a completely different circle must not match."""
        ref = _circle_cloud(300, 2.0, np.array([0.0, 0.0]))
        probe = _circle_cloud(300, 2.0, np.array([20.0, 20.0]))  # far away
        assert not evaluate_target_match(
            probe, ref, metric="nn_percentile", tolerance=1.0, nn_percentile=90.0
        )

    def test_tolerates_density_mismatch(self):
        """nn_percentile should be robust when probe cloud is much sparser than ref."""
        ref = _circle_cloud(1000, 2.0, np.array([0.0, 0.0]))
        probe = _circle_cloud(30, 2.0, np.array([0.0, 0.0]))  # sparse probe, same orbit
        assert evaluate_target_match(
            probe, ref, metric="nn_percentile", tolerance=0.6, nn_percentile=90.0
        )

    def test_partially_overlapping_cloud_uses_percentile(self):
        """When 80 % of probe is on-target but 20 % is elsewhere, p90 should still reject."""
        rng = np.random.default_rng(7)
        ref = _circle_cloud(300, 2.0, np.array([0.0, 0.0]))
        on_target = _circle_cloud(240, 2.0, np.array([0.0, 0.0]))
        off_target = _circle_cloud(60, 0.5, np.array([30.0, 30.0]))  # far outliers
        probe = np.vstack([on_target, off_target])
        # p90 covers the off-target points → should fail with tight tolerance
        assert not evaluate_target_match(
            probe, ref, metric="nn_percentile", tolerance=1.0, nn_percentile=90.0
        )
        # p75 stays within the 80% on-target → should pass
        assert evaluate_target_match(
            probe, ref, metric="nn_percentile", tolerance=1.0, nn_percentile=75.0
        )

    def test_large_cloud_subsampling_path(self):
        """Clouds > 2 000 points must be sub-sampled and still give correct answer."""
        ref = _circle_cloud(5000, 2.0, np.array([0.0, 0.0]))
        probe = _circle_cloud(5000, 2.0, np.array([0.0, 0.0]))
        # Should match because both are from the same underlying orbit
        assert evaluate_target_match(
            probe, ref, metric="nn_percentile", tolerance=0.5, nn_percentile=90.0
        )

    def test_3d_cloud(self):
        """Metric must work in 3-D (torus-like attractor proxy)."""
        rng = np.random.default_rng(3)
        angles = rng.uniform(0, 2 * np.pi, 300)
        ref = np.column_stack([
            (3.0 + np.cos(angles)) * np.cos(angles),
            (3.0 + np.cos(angles)) * np.sin(angles),
            np.sin(angles),
        ])
        probe = ref + rng.normal(0, 0.05, ref.shape)  # tiny noise
        assert evaluate_target_match(
            probe, ref, metric="nn_percentile", tolerance=0.3, nn_percentile=90.0
        )

    def test_empty_arrays_return_false(self):
        ref = _circle_cloud(50, 1.0, np.array([0.0, 0.0]))
        assert not evaluate_target_match(
            np.empty((0, 2)), ref, metric="nn_percentile", tolerance=0.5
        )
        assert not evaluate_target_match(
            ref, np.empty((0, 2)), metric="nn_percentile", tolerance=0.5
        )

    def test_unknown_metric_raises(self):
        ref = _circle_cloud(10, 1.0, np.array([0.0, 0.0]))
        with pytest.raises(ValueError, match="Unknown target_match metric"):
            evaluate_target_match(ref, ref, metric="hausdorff", tolerance=0.5)
