from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.workflows.biased_chua import _extended_radius_blocks


def test_extended_probe_schedule_is_radius_major_and_seed_stable() -> None:
    equilibria = {
        "E0": np.array([0.0, 0.0, 0.0]),
        "E+": np.array([1.0, 0.0, -1.0]),
        "E-": np.array([-1.0, 0.0, 1.0]),
    }
    blocks = _extended_radius_blocks(
        equilibria,
        [(1.0e-3, 4), (3.0e-3, 6), (1.0e-2, 8)],
        random_seed=42,
    )

    assert [[entry[1] for entry in block] for block in blocks] == [
        ["E0", "E+", "E-"],
        ["E0", "E+", "E-"],
        ["E0", "E+", "E-"],
    ]
    assert [[entry[4] for entry in block] for block in blocks] == [
        [1.0e-3] * 3,
        [3.0e-3] * 3,
        [1.0e-2] * 3,
    ]
    assert [[entry[6] for entry in block] for block in blocks] == [
        [42, 142, 242],
        [52, 152, 252],
        [62, 162, 262],
    ]


def test_extended_probe_schedule_rejects_noncausal_radius_order() -> None:
    equilibria = {"E0": np.zeros(3)}

    with pytest.raises(ValueError, match="strictly increasing"):
        _extended_radius_blocks(
            equilibria,
            [(1.0e-2, 8), (1.0e-3, 4)],
            random_seed=42,
        )
