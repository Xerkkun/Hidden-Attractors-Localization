from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import pytest

from hidden_attractors._time_grid import checked_array_capacity, exact_fixed_step_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_clean_package_import_does_not_load_numerical_stacks() -> None:
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
        "import hidden_attractors; "
        "import hidden_attractors.analysis; "
        "import hidden_attractors.fractional; "
        "import hidden_attractors.integrations; "
        "import hidden_attractors.workflows; "
        "forbidden={'numba','scipy','hidden_attractors.analysis.lyapunov',"
        "'hidden_attractors.fractional.tempered_fast_history',"
        "'hidden_attractors.integrations.fractional_c',"
        "'hidden_attractors.workflows.attractor_only'}; "
        "loaded=sorted(forbidden.intersection(sys.modules)); "
        "assert not loaded, loaded; "
        "print('LAZY_IMPORT_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LAZY_IMPORT_OK"


def test_top_level_lazy_resolution_preserves_api_tier() -> None:
    import hidden_attractors as ha

    assert ha.ChuaParameters.__name__ == "ChuaParameters"
    assert ha.get_tier(ha.ChuaParameters) == ha.STABLE


def test_same_named_submodule_import_preserves_public_callable() -> None:
    import importlib

    import hidden_attractors as ha
    import hidden_attractors.analysis as analysis
    import hidden_attractors.fractional as fractional

    entropy_module = importlib.import_module(
        "hidden_attractors.analysis.permutation_entropy"
    )
    tempered_module = importlib.import_module(
        "hidden_attractors.fractional.tempered_convolution_quadrature"
    )

    assert analysis.permutation_entropy is entropy_module.permutation_entropy
    assert ha.permutation_entropy is entropy_module.permutation_entropy
    assert (
        fractional.tempered_convolution_quadrature
        is tempered_module.tempered_convolution_quadrature
    )


def test_fixed_step_count_rejects_overflowing_ratio() -> None:
    with pytest.raises(ValueError, match="t_final / h must be finite"):
        exact_fixed_step_count(
            math.nextafter(0.0, 1.0),
            1.0,
            caller="test",
        )


def test_fixed_step_count_rejects_finite_but_unrepresentable_ratio() -> None:
    with pytest.raises(ValueError, match="exceeding the supported limit"):
        exact_fixed_step_count(1.0e-300, 1.0, caller="test")


def test_array_capacity_rejects_platform_size_overflow() -> None:
    with pytest.raises(ValueError, match="exceeds platform limits"):
        checked_array_capacity(
            (sys.maxsize, 2),
            float,
            caller="test",
        )


def test_direct_integer_solver_rejects_noninteger_horizon() -> None:
    import numpy as np

    from hidden_attractors.solvers.integer import efork_q1_integrate

    with pytest.raises(ValueError, match="integer number of fixed steps"):
        efork_q1_integrate(
            lambda state: -state,
            np.array([1.0]),
            t_final=1.0,
            h=0.3,
        )


def test_integer_lyapunov_rejects_noninteger_burn_grid() -> None:
    import numpy as np

    from hidden_attractors.analysis.lyapunov import integer_lyapunov_exponents

    with pytest.raises(ValueError, match="integer number of fixed steps"):
        integer_lyapunov_exponents(
            lambda state: -state,
            lambda _state: np.array([[-1.0]]),
            np.array([1.0]),
            h=0.3,
            t_final=0.9,
            t_burn=1.0,
        )
