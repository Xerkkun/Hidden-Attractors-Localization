from __future__ import annotations

from hidden_attractors.capabilities import get_capability, list_capabilities


def test_capability_catalog_separates_integer_and_fractional_support() -> None:
    recurrence = get_capability("recurrence_quantification")
    assert recurrence.integer_status == "implemented"
    assert recurrence.fractional_status == "implemented"
    assert recurrence.trajectory_based

    periodic = get_capability("periodic_orbits")
    assert periodic.integer_status == "planned"
    assert periodic.fractional_status == "research_required"

    basin_entropy = get_capability("basin_entropy")
    assert basin_entropy.integer_status == "implemented"
    assert basin_entropy.fractional_status == "implemented"
    assert basin_entropy.trajectory_based

    adapters = get_capability("complexity_measure_adapters")
    assert adapters.integer_status == "implemented"
    assert adapters.fractional_status == "implemented"
    assert adapters.trajectory_based

    dimension = get_capability("correlation_dimension")
    assert dimension.integer_status == "implemented"
    assert dimension.fractional_status == "implemented"
    assert dimension.trajectory_based
    assert dimension.backend == "c/numba"

    contract = get_capability("trajectory_analysis_contract")
    assert contract.integer_status == "implemented"
    assert contract.fractional_status == "implemented"
    assert contract.backend == "hafo"

    permutation = get_capability("permutation_entropy")
    assert permutation.integer_status == "implemented"
    assert permutation.fractional_status == "implemented"
    assert permutation.trajectory_based
    assert permutation.backend == "c/numba"

    multi_term = get_capability("multi_term_caputo_l1")
    assert multi_term.integer_status == "implemented_limit"
    assert multi_term.fractional_status == "implemented"
    assert not multi_term.trajectory_based
    assert multi_term.backend == "numba/python"
    assert "never normalized" in multi_term.notes

    tempered_cq = get_capability("tempered_convolution_quadrature")
    assert tempered_cq.integer_status == "not_applicable"
    assert tempered_cq.fractional_status == "implemented"
    assert tempered_cq.backend == "numba/fft"
    assert "offline FFT" in tempered_cq.notes
    assert "Symbol-shift CQ" in tempered_cq.notes
    assert "remains planned" in tempered_cq.notes
    assert "recurrent FBDF1/GNGF2 fast history" in tempered_cq.notes
    assert "separate experimental capability" in tempered_cq.notes

    alignment = get_capability("sali_gali_alignment_indices")
    assert alignment.integer_status == "implemented"
    assert alignment.fractional_status == "research_required"
    assert not alignment.trajectory_based
    assert alignment.backend == "numpy/numba/scipy"
    assert "history-space tangent theory" in alignment.notes

    covariant = get_capability("covariant_lyapunov_vectors")
    assert covariant.integer_status == "implemented"
    assert covariant.fractional_status == "research_required"
    assert not covariant.trajectory_based
    assert covariant.backend == "numpy/numba/scipy"
    assert "history-space tangent theory" in covariant.notes


def test_capability_catalog_records_upstream_inspiration_without_backend_dependency() -> None:
    compiled = get_capability("numba_compiled_flow_map")
    assert "pynamicalsys" in compiled.inspiration
    assert compiled.backend == "numba"
    assert list_capabilities(category="chaos")
