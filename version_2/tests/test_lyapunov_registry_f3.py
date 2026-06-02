"""Registry contract for F3 cloned dynamics."""

from hidden_attractors.analysis.lyapunov_methods import LYAPUNOV_METHODS


def test_published_gs_registry_contract() -> None:
    info = LYAPUNOV_METHODS["fractional_cloned_dynamics_abm_gs_published"]
    assert info.implemented is True
    assert info.validated is False
    assert info.validated_against_published_benchmarks is False
    assert info.requires_jacobian is False
    assert info.method_type == "cloned_dynamics"
    assert info.memory_protocol == "published_block_restart"
    assert info.benchmark_status == "published_benchmarks_pending_discrepancy"
    assert info.supports_q_equal_1 is True
    assert info.supports_incommensurate is True


def test_qr_registry_contract() -> None:
    info = LYAPUNOV_METHODS["fractional_cloned_dynamics_abm_qr"]
    assert info.implemented is True
    assert info.validated is False
    assert info.validated_against_published_benchmarks is False
    assert info.requires_jacobian is False
    assert info.orthonormalization == "qr"
    assert info.benchmark_status == "internal_variant_pending"


def test_fractional_variational_validation_status_is_unchanged() -> None:
    assert LYAPUNOV_METHODS["fractional_variational_abm_qr"].validated is False
