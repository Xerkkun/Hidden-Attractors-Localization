from __future__ import annotations

import numpy as np

from tools.chua_candidate_extended_hiddenness import (
    build_directions,
    build_probe_tasks,
)


def test_extended_direction_cloud_has_requested_density_and_unit_norms() -> None:
    directions = build_directions(96)
    assert len(directions) == 96
    assert sum(row["direction_family"] == "axis" for row in directions) == 6
    assert sum(row["direction_family"] == "fibonacci" for row in directions) == 90
    vectors = np.asarray([row["direction_vector"] for row in directions], dtype=float)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=0.0, atol=1.0e-12)
    assert len({row["direction"] for row in directions}) == 96


def test_extended_probe_plan_covers_all_equilibria_radii_and_directions() -> None:
    equilibria = {
        "E0": [0.0, 0.0, 0.0],
        "E+": [1.0, 0.0, -1.0],
        "E-": [-1.0, 0.0, 1.0],
    }
    radii = [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 1.0e-2]
    samples_per_radius = [96] * len(radii)
    tasks = build_probe_tasks(equilibria, radii, samples_per_radius)
    assert len(tasks) == 3 * 6 * 96
    assert len({task["probe_id"] for task in tasks}) == len(tasks)
    for equilibrium in equilibria:
        local = [task for task in tasks if task["equilibrium"] == equilibrium]
        assert len(local) == 6 * 96
        for radius in radii:
            assert sum(np.isclose(task["radius"], radius) for task in local) == 96
