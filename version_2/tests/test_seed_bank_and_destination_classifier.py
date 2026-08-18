"""Focused contracts for the geometric/topological campaign seed and destination layers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from hidden_attractors.seed_bank import (
    SEED_BANK_CSV_FIELDS,
    SeedRecord,
    SymmetryTransform,
    build_seed_bank,
    load_seed_bank,
    write_seed_bank,
)
from hidden_attractors.verification.destination_classifier import (
    DESTINATION_CLASSIFIER_SCHEMA_VERSION,
    DESTINATION_LABELS,
    DestinationClassifierContract,
    classify_destination,
)


def _integer_seed(seed_id: str, route: str, state: tuple[float, ...], **kwargs) -> SeedRecord:
    return SeedRecord(
        seed_id=seed_id,
        system_id="fixture",
        parameter_set_id="p0",
        route=route,
        state=state,
        order_kind="integer",
        q=1.0,
        **kwargs,
    )


def test_seed_record_adapts_current_unified_mapping_without_claim_inflation() -> None:
    record = SeedRecord.from_mapping(
        {
            "candidate_id": "df-1",
            "family": "lure_classical_centered",
            "q": 0.99,
            "x0": [1.0, 0.0, -1.0],
            "harmonic_residual": 1.0e-8,
            "A": 2.0,
            "omega": 3.0,
        },
        system_id="chua-nonsmooth",
        order_kind="caputo",
        lower_terminal=0.0,
        initial_time=0.0,
    )

    assert record.route == "describing_function"
    assert record.state == (1.0, 0.0, -1.0)
    assert record.generation_residual == pytest.approx(1.0e-8)
    assert record.metadata["source_family"] == "lure_classical_centered"
    assert record.metadata["A"] == pytest.approx(2.0)


def test_seed_record_requires_explicit_caputo_lower_terminal_and_history_reference() -> None:
    with pytest.raises(ValueError, match="lower_terminal"):
        SeedRecord(
            seed_id="fractional",
            system_id="fixture",
            route="fractional_perpetual_point",
            state=(1.0, 2.0),
            order_kind="caputo",
            q=0.9,
        )
    with pytest.raises(ValueError, match="history_reference"):
        SeedRecord(
            seed_id="history",
            system_id="fixture",
            route="continuation",
            state=(1.0, 2.0),
            order_kind="caputo",
            q=0.9,
            lower_terminal=0.0,
            initial_time=10.0,
            initialization_kind="continued_history",
        )


def test_seed_mapping_respects_explicit_caputo_order_at_q_one() -> None:
    record = SeedRecord.from_mapping(
        {
            "seed_id": "caputo-q1",
            "system_id": "fixture",
            "route": "manual",
            "state": [1.0],
            "order_kind": "caputo",
            "q": 1.0,
            "lower_terminal": 0.0,
            "initial_time": 0.0,
        }
    )

    assert record.order_kind == "caputo"


def test_seed_metadata_rejects_path_and_other_non_json_objects() -> None:
    with pytest.raises(TypeError, match="non-JSON"):
        _integer_seed("bad-metadata", "manual", (1.0,), metadata={"path": Path("x")})


def test_seed_bank_deduplicates_across_routes_but_preserves_provenance() -> None:
    records = (
        _integer_seed(
            "df",
            "describing_function",
            (1.0, 2.0),
            generation_residual=1.0e-7,
        ),
        _integer_seed(
            "machado",
            "machado",
            (1.0 + 1.0e-9, 2.0 - 1.0e-9),
            generation_residual=1.0e-6,
        ),
    )
    bank = build_seed_bank(records, coordinate_scale=(1.0, 1.0))

    assert len(bank.records) == 2
    assert len(bank.representatives) == 1
    assert {item.record.route for item in bank.memberships} == {
        "describing_function",
        "machado",
    }
    duplicate = next(item for item in bank.memberships if not item.is_representative)
    assert duplicate.duplicate_of == "df"


def test_seed_bank_uses_declared_scale_and_symmetry_orbits() -> None:
    positive = _integer_seed("positive", "perpetual_point", (1000.0, 1.0e-3))
    negative = _integer_seed("negative", "critical_surface", (-1000.0, -1.0e-3))
    bank = build_seed_bank(
        (positive, negative),
        coordinate_scale=(1000.0, 1.0e-3),
        symmetries=(SymmetryTransform.inversion(2),),
        symmetry_group_is_complete=True,
    )

    assert len(bank.representatives) == 1
    duplicate = next(item for item in bank.memberships if not item.is_representative)
    assert duplicate.matched_symmetry == "inversion"
    assert duplicate.normalized_distance == pytest.approx(0.0)


def test_seed_bank_wraps_declared_periodic_coordinate_for_pll() -> None:
    first = _integer_seed("phase-a", "edge_tracking", (0.1, 0.5))
    wrapped = _integer_seed("phase-b", "edge_tracking", (0.1 + 2.0 * np.pi, 0.5))
    bank = build_seed_bank(
        (first, wrapped),
        coordinate_scale=(1.0, 1.0),
        periodic_coordinates={0: 2.0 * np.pi},
    )

    assert len(bank.representatives) == 1
    duplicate = next(item for item in bank.memberships if not item.is_representative)
    assert duplicate.normalized_distance == pytest.approx(0.0, abs=1.0e-14)


def test_seed_bank_requires_explicit_complete_finite_symmetry_group() -> None:
    seed = _integer_seed("seed", "manual", (1.0, 2.0))
    with pytest.raises(ValueError, match="complete finite group"):
        build_seed_bank(
            (seed,),
            coordinate_scale=(1.0, 1.0),
            symmetries=(SymmetryTransform.inversion(2),),
        )


def test_seed_bank_never_merges_distinct_fractional_histories() -> None:
    common = dict(
        system_id="fractional-fixture",
        route="continuation",
        state=(1.0, 2.0),
        order_kind="caputo",
        q=0.9,
        lower_terminal=0.0,
        initialization_kind="continued_history",
        history_coverage=(0.0, 10.0),
        initial_time=10.0,
    )
    first = SeedRecord(seed_id="h1", history_reference="history-a.csv", **common)
    second = SeedRecord(seed_id="h2", history_reference="history-b.csv", **common)
    bank = build_seed_bank((first, second), coordinate_scale=(1.0, 1.0))

    assert len(bank.representatives) == 2


def test_caputo_history_coverage_must_cover_terminal_through_initial_time() -> None:
    with pytest.raises(ValueError, match="start at/before"):
        SeedRecord(
            seed_id="late-history",
            system_id="fixture",
            route="continuation",
            state=(1.0,),
            order_kind="caputo",
            q=0.9,
            lower_terminal=0.0,
            initial_time=10.0,
            initialization_kind="continued_history",
            history_reference="history.csv",
            history_coverage=(1.0, 10.0),
        )
    with pytest.raises(ValueError, match="end at/after"):
        SeedRecord(
            seed_id="short-history",
            system_id="fixture",
            route="continuation",
            state=(1.0,),
            order_kind="caputo",
            q=0.9,
            lower_terminal=0.0,
            initial_time=10.0,
            initialization_kind="continued_history",
            history_reference="history.csv",
            history_coverage=(0.0, 9.0),
        )


def test_history_based_caputo_seed_rejects_state_only_symmetry_deduplication() -> None:
    history = SeedRecord(
        seed_id="history",
        system_id="fixture",
        route="continuation",
        state=(1.0,),
        order_kind="caputo",
        q=0.9,
        lower_terminal=0.0,
        initial_time=10.0,
        initialization_kind="continued_history",
        history_reference="history.csv",
        history_coverage=(0.0, 10.0),
    )
    with pytest.raises(ValueError, match="hereditary state"):
        build_seed_bank(
            (history,),
            coordinate_scale=(1.0,),
            symmetries=(SymmetryTransform.inversion(1),),
            symmetry_group_is_complete=True,
        )


def test_seed_bank_writes_matching_json_and_stable_csv(tmp_path) -> None:
    bank = build_seed_bank(
        (_integer_seed("manual", "manual", (1.0, -2.0)),),
        coordinate_scale=(2.0, 4.0),
        periodic_coordinates={0: 2.0 * np.pi},
        symmetries=(SymmetryTransform.inversion(2),),
        symmetry_group_is_complete=True,
    )
    paths = write_seed_bank(tmp_path, bank)

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["n_records"] == 1
    assert payload["n_representatives"] == 1
    assert "not evidence" in payload["scientific_scope"]
    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert tuple(reader.fieldnames or ()) == SEED_BANK_CSV_FIELDS
    assert rows[0]["state"] == "1;-2"
    assert rows[0]["priority"] == "100"
    assert rows[0]["matched_symmetry"] == "identity"
    restored_json = load_seed_bank(paths["json"])
    restored_csv = load_seed_bank(paths["csv"])
    assert restored_json.to_dict() == bank.to_dict()
    assert restored_csv.to_dict() == bank.to_dict()


def test_seed_bank_emits_representative_before_lower_identifier_duplicate() -> None:
    preferred = _integer_seed(
        "z-preferred",
        "manual",
        (1.0,),
        priority=0,
    )
    duplicate = _integer_seed(
        "a-duplicate",
        "describing_function",
        (1.0,),
        priority=100,
    )
    bank = build_seed_bank((duplicate, preferred), coordinate_scale=(1.0,))

    assert [item.record.seed_id for item in bank.memberships] == [
        "z-preferred",
        "a-duplicate",
    ]
    assert bank.memberships[0].is_representative


def _contract(**kwargs) -> DestinationClassifierContract:
    defaults = dict(
        burn_time=20.0,
        coordinate_scale=(1.0, 1.0),
        min_tail_samples=128,
        max_cloud_points=250,
    )
    defaults.update(kwargs)
    return DestinationClassifierContract(**defaults)


def test_destination_classifier_resolves_equilibrium_from_tail_not_final_point_only() -> None:
    times = np.linspace(0.0, 40.0, 4001)
    states = np.column_stack((np.exp(-times), -2.0 * np.exp(-times)))
    result = classify_destination(
        times,
        states,
        contract=_contract(),
        equilibria={"origin": np.zeros(2)},
    )

    assert result.label == "equilibrium"
    assert result.destination_id == "equilibrium:origin"
    assert result.metrics["closest_equilibrium_q95_distance_norm"] < 1.0e-3


def test_destination_classifier_detects_near_periodic_integer_motion() -> None:
    times = np.arange(0.0, 100.0, 0.01)
    states = np.column_stack(
        (
            np.sin(2.0 * np.pi * 0.5 * times),
            np.cos(2.0 * np.pi * 0.5 * times),
        )
    )
    result = classify_destination(times, states, contract=_contract())

    assert result.label == "periodic"
    assert result.subtype == "near_periodic_integer_flow"
    assert result.metrics["periodic_gate_passed"] is True


def test_caputo_periodic_label_is_explicitly_projected_and_finite_time() -> None:
    times = np.arange(0.0, 100.0, 0.01)
    states = np.column_stack(
        (
            np.sin(2.0 * np.pi * 0.25 * times),
            np.cos(2.0 * np.pi * 0.25 * times),
        )
    )
    result = classify_destination(
        times,
        states,
        contract=_contract(order_kind="caputo"),
    )

    assert result.label == "periodic"
    assert result.subtype == "projected_near_periodic_caputo"
    assert any("not an exact periodic" in warning for warning in result.scientific_warnings)


def test_destination_classifier_detects_stationary_recurrent_cloud() -> None:
    rng = np.random.default_rng(20260811)
    times = np.arange(0.0, 120.0, 0.01)
    states = rng.uniform(-1.0, 1.0, size=(times.size, 2))
    result = classify_destination(times, states, contract=_contract())

    assert result.label == "recurrent"
    assert result.metrics["stationarity_gate_passed"] is True
    assert result.metrics["recurrence_rate"] > 0.0
    assert result.metrics["recurrence_theiler_samples"] == 10


def test_excursion_return_fraction_counts_unique_anchors_and_is_bounded() -> None:
    rng = np.random.default_rng(20260811)
    times = np.arange(0.0, 120.0, 0.01)
    states = rng.uniform(-1.0, 1.0, size=(times.size, 2))
    result = classify_destination(
        times,
        states,
        contract=_contract(recurrence_radius=0.3),
    )
    metrics = result.metrics

    assert result.schema_version == DESTINATION_CLASSIFIER_SCHEMA_VERSION == "1.1"
    assert metrics["excursion_return_pair_count"] > metrics["recurrence_sample_count"]
    assert metrics["excursion_return_count"] == metrics["excursion_return_pair_count"]
    assert 0.0 <= metrics["excursion_return_fraction"] <= 1.0
    assert metrics["excursion_return_fraction"] == pytest.approx(
        metrics["excursion_return_anchor_count"] / metrics["recurrence_sample_count"]
    )
    assert metrics["excursion_return_pairs_per_sample"] == pytest.approx(
        metrics["excursion_return_pair_count"] / metrics["recurrence_sample_count"]
    )
    assert metrics["excursion_return_pairs_per_sample"] > 1.0
    assert metrics["excursion_return_mean_multiplicity"] == pytest.approx(
        metrics["excursion_return_pair_count"]
        / metrics["excursion_return_anchor_count"]
    )


def test_theiler_window_prevents_smooth_monotone_neighbors_from_faking_recurrence() -> None:
    times = np.arange(0.0, 100.0, 0.01)
    states = np.column_stack((0.001 * times, np.zeros_like(times)))
    result = classify_destination(
        times,
        states,
        contract=_contract(),
    )

    assert result.label == "transient"
    assert result.label != "recurrent"
    assert result.metrics["excursion_return_count"] == 0
    assert result.metrics["excursion_return_pair_count"] == 0
    assert result.metrics["excursion_return_anchor_count"] == 0
    assert result.metrics["excursion_return_fraction"] == 0.0
    assert result.metrics["excursion_return_pairs_per_sample"] == 0.0
    assert result.metrics["excursion_return_mean_multiplicity"] == 0.0


def test_destination_classifier_separates_escape_and_unsettled_transient() -> None:
    times = np.arange(0.0, 100.0, 0.01)
    escaping = np.column_stack((0.2 * times, np.zeros_like(times)))
    escape = classify_destination(
        times,
        escaping,
        contract=_contract(burn_time=0.0, divergence_radius=10.0),
        integration_status="event_escape",
    )
    drifting = np.column_stack((0.01 * times, 0.002 * times))
    transient = classify_destination(times, drifting, contract=_contract())

    assert escape.label == "escape"
    assert transient.label == "transient"


def test_explicit_escape_with_absolute_radius_precedes_short_tail_gate() -> None:
    times = np.linspace(0.0, 1.0, 20)
    states = np.column_stack((20.0 * times, np.zeros_like(times)))
    result = classify_destination(
        times,
        states,
        contract=DestinationClassifierContract(
            burn_time=0.0,
            coordinate_scale=None,
            divergence_radius=10.0,
            divergence_radius_kind="absolute",
            min_tail_samples=128,
        ),
        integration_status="event_escape",
    )

    assert result.label == "escape"
    assert result.subtype == "explicit_solver_escape_with_finite_radius_evidence"


def test_inferred_scale_cannot_certify_a_non_escape_campaign_destination() -> None:
    times = np.arange(0.0, 100.0, 0.01)
    states = np.column_stack((np.sin(times), np.cos(times)))
    result = classify_destination(
        times,
        states,
        contract=DestinationClassifierContract(burn_time=20.0),
    )

    assert result.label == "ambiguous"
    assert result.subtype == "inferred_scale_cannot_support_campaign_destination"


def test_contract_normalizes_numeric_strings_and_rejects_non_numeric_thresholds() -> None:
    contract = DestinationClassifierContract(
        burn_time="2.5",
        coordinate_scale=("1", "2"),
        min_tail_samples="128",
        recurrence_theiler_samples="12",
    )
    assert contract.burn_time == pytest.approx(2.5)
    assert contract.min_tail_samples == 128
    assert contract.recurrence_theiler_samples == 12
    with pytest.raises(TypeError, match="real numeric"):
        DestinationClassifierContract(
            burn_time=0.0,
            coordinate_scale=(1.0,),
            recurrence_radius="not-a-number",
        )


def test_low_confidence_transient_is_demoted_to_ambiguous() -> None:
    times = np.arange(0.0, 100.0, 0.01)
    states = np.column_stack((0.01 * times, 0.002 * times))
    result = classify_destination(
        times,
        states,
        contract=_contract(minimum_confidence=1.0),
    )

    assert result.label == "ambiguous"
    assert result.subtype == "unsettled_without_sufficient_transient_margin"


def test_destination_classifier_marks_insufficient_or_nonfinite_data_ambiguous() -> None:
    short_times = np.linspace(0.0, 1.0, 50)
    short_states = np.zeros((50, 2))
    insufficient = classify_destination(
        short_times,
        short_states,
        contract=_contract(burn_time=0.0),
    )
    states = np.zeros((200, 2))
    states[-1, 0] = np.nan
    nonfinite = classify_destination(
        np.linspace(0.0, 2.0, 200),
        states,
        contract=_contract(burn_time=0.0),
    )

    assert insufficient.label == "ambiguous"
    assert nonfinite.label == "ambiguous"
    assert insufficient.is_ambiguous and nonfinite.is_ambiguous
    json.dumps(insufficient.to_dict(), allow_nan=False)
    json.dumps(nonfinite.to_dict(), allow_nan=False)
    assert set(DESTINATION_LABELS) == {
        "equilibrium",
        "periodic",
        "recurrent",
        "escape",
        "transient",
        "ambiguous",
    }


def test_destination_classifier_assigns_specific_reference_for_edge_tracking() -> None:
    times = np.arange(0.0, 100.0, 0.01)
    states = np.column_stack(
        (
            np.sin(2.0 * np.pi * 0.5 * times),
            np.cos(2.0 * np.pi * 0.5 * times),
        )
    )
    references = {
        "cycle_a": states[times >= 20.0],
        "cycle_b": states[times >= 20.0] + np.array([5.0, 5.0]),
    }
    result = classify_destination(
        times,
        states,
        contract=_contract(),
        references=references,
    )

    assert result.label == "periodic"
    assert result.destination_id == "reference:cycle_a"
    assert result.edge_label == "reference:cycle_a"


def test_periodic_coordinate_wraps_equilibrium_and_reference_distances_for_pll() -> None:
    times = np.arange(0.0, 100.0, 0.01)
    decaying = np.exp(-times)
    equilibrium_states = np.column_stack(
        (np.full(times.size, 0.1 + 2.0 * np.pi), decaying)
    )
    contract = _contract(periodic_coordinates={0: 2.0 * np.pi})
    equilibrium = classify_destination(
        times,
        equilibrium_states,
        contract=contract,
        equilibria={"pll_eq": np.array([0.1, 0.0])},
    )

    phase = np.mod(2.0 * np.pi * 0.5 * times, 2.0 * np.pi)
    cycle = np.column_stack((phase, np.sin(phase)))
    references = {
        "pll_cycle": cycle[times >= 20.0] + np.array([2.0 * np.pi, 0.0]),
        "far": np.column_stack((phase[times >= 20.0], np.sin(phase[times >= 20.0]) + 5.0)),
    }
    matched = classify_destination(
        times,
        cycle,
        contract=contract,
        references=references,
    )

    assert equilibrium.label == "equilibrium"
    assert matched.destination_id == "reference:pll_cycle"
    assert matched.metrics["metric_space"] == "cylindrical_product_embedding"
