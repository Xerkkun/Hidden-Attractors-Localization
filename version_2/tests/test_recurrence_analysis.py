from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.analysis.recurrence import (
    delay_embedding,
    recurrence_matrix,
    recurrence_quantification,
)


def test_delay_embedding_handles_scalar_and_multivariate_samples() -> None:
    scalar = delay_embedding(np.arange(6, dtype=float), dimension=3, delay=2)
    assert np.array_equal(scalar, [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])

    multi = delay_embedding(
        np.column_stack((np.arange(4), np.arange(4) + 10)),
        dimension=2,
        delay=1,
    )
    assert np.array_equal(multi[0], [0.0, 10.0, 1.0, 11.0])


def test_recurrence_matrix_is_symmetric_and_excludes_identity_line() -> None:
    points = np.array([[0.0], [0.1], [1.0]])
    matrix = recurrence_matrix(points, radius=0.11)
    assert np.array_equal(matrix, matrix.T)
    assert not np.any(np.diag(matrix))
    assert matrix[0, 1] and not matrix[0, 2]


def test_periodic_sequence_has_deterministic_recurrence_lines() -> None:
    points = np.tile([[0.0], [1.0]], (12, 1))
    result = recurrence_quantification(points, radius=0.0, return_matrix=True)
    assert result.recurrence_rate > 0.45
    assert result.determinism > 0.9
    assert result.longest_diagonal >= 20
    assert result.recurrence_matrix is not None
    assert result.status == "finite_numerical_diagnostic"


def test_recurrence_memory_guard_is_explicit() -> None:
    with pytest.raises(MemoryError, match="above max_matrix_bytes"):
        recurrence_matrix(np.zeros((20, 1)), radius=0.0, max_matrix_bytes=399)

