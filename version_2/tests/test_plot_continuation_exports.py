"""Regression tests for the public continuation plotting outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hidden_attractors.plotting import (
    plot_continuation_eta,
    plot_continuation_tracking,
    plot_harmonic_residual_map,
    plot_nyquist_transfer,
)


def _steps() -> list[dict]:
    time = np.linspace(0.0, 2.0, 80)
    rows: list[dict] = []
    for index, eta in enumerate((0.0, 0.5, 1.0)):
        trajectory = np.column_stack(
            (
                time,
                np.sin(time + eta),
                np.cos(time + eta),
                0.5 * np.sin(2.0 * time + eta),
            )
        )
        x_out = trajectory[-1, 1:]
        rows.append(
            {
                "lambda_value": eta,
                "x_in": np.array([eta, 0.0, 0.0]),
                "x_out": x_out,
                "x_out_norm": float(np.linalg.norm(x_out)),
                "trajectory": trajectory,
                "status": "ok" if index != 1 else "converged_equilibrium_early",
            }
        )
    return rows


def _isolate_figure_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import hidden_attractors.plotting.export as export_module
    import hidden_attractors.plotting.manifest as manifest_module

    root = tmp_path / "library_figures"
    monkeypatch.setattr(export_module, "LIBRARY_FIGURES_ROOT", root)
    monkeypatch.setattr(manifest_module, "LIBRARY_FIGURES_ROOT", root)


def _capture_figure_text(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    import hidden_attractors.plotting.export as export_module

    captured: dict[str, list[str]] = {}

    def capture(fig, output_path, _figure_type):
        text: list[str] = []
        for axis in fig.axes:
            text.extend((axis.get_title(), axis.get_xlabel(), axis.get_ylabel()))
            if hasattr(axis, "get_zlabel"):
                text.append(axis.get_zlabel())
            text.extend(line.get_label() for line in axis.lines)
            text.extend(collection.get_label() for collection in axis.collections)
        captured[Path(output_path).name] = text

    monkeypatch.setattr(export_module, "intercept_and_export_path", capture)
    return captured


@pytest.mark.unit
def test_plot_continuation_eta_exports_norm_and_amplitude(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_figure_store(monkeypatch, tmp_path)
    output_dir = tmp_path / "eta"

    plot_continuation_eta(
        _steps(),
        {"system_id": "synthetic_test"},
        output_dir,
    )

    assert (output_dir / "figures" / "continuation_norm_vs_eta.png").is_file()
    assert (output_dir / "figures" / "continuation_amplitude_vs_eta.png").is_file()


@pytest.mark.unit
def test_plot_continuation_tracking_exports_norm_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_figure_store(monkeypatch, tmp_path)
    output_dir = tmp_path / "tracking"

    plot_continuation_tracking(
        _steps(),
        {"system_id": "synthetic_test"},
        output_dir,
    )

    assert (output_dir / "figures" / "continuation_tracking_norm.png").is_file()
    assert (output_dir / "figures" / "continuation_tracking_status.png").is_file()


@pytest.mark.unit
def test_all_continuation_plot_labels_use_public_lambda_parameter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_figure_text(monkeypatch)

    plot_continuation_eta(
        _steps(),
        {"system_id": "label_contract"},
        tmp_path / "eta",
    )
    plot_continuation_tracking(
        _steps(),
        {"system_id": "label_contract"},
        tmp_path / "tracking",
    )

    assert {
        "continuation_norm_vs_eta.png",
        "continuation_amplitude_vs_eta.png",
        "continuation_first_last_comparison.png",
        "continuation_first_last_projections.png",
        "continuation_timeseries_comparison_x.png",
        "continuation_progression.png",
        "continuation_tracking_norm.png",
        "continuation_tracking_status.png",
    } <= captured.keys()

    visible_text = "\n".join(
        item
        for figure_text in captured.values()
        for item in figure_text
        if isinstance(item, str)
    )
    assert r"\eta" not in visible_text
    assert r"\lambda" in visible_text
    assert r"\lambda=0" in visible_text
    assert r"\lambda=1" in visible_text


@pytest.mark.unit
@pytest.mark.parametrize(
    ("contract", "expected_target", "expected_target_text", "transfer_name"),
    [
        ({}, 0.5, "1/k", "W_{\\mathrm{report}}"),
        (
            {
                "transfer_convention": "standard",
                "harmonic_condition": "1_minus_WN",
            },
            0.5,
            "1/k",
            "W_{\\mathrm{report}}",
        ),
        (
            {
                "transfer_convention": "opposite_sign",
                "harmonic_condition": "1_plus_WN",
            },
            -0.5,
            "-1/k",
            "W_{\\mathrm{code}}",
        ),
    ],
)
def test_nyquist_plot_respects_transfer_contract_and_keeps_both_half_planes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, str],
    expected_target: float,
    expected_target_text: str,
    transfer_name: str,
) -> None:
    import hidden_attractors.plotting.export as export_module

    captured: dict[str, dict[str, object]] = {}

    def capture(fig, output_path, _figure_type):
        name = Path(output_path).name
        if name == "transfer_nyquist.png":
            axis = fig.axes[0]
            captured[name] = {
                "candidate": np.asarray(axis.collections[0].get_offsets(), dtype=float),
                "ylim": axis.get_ylim(),
                "legend_labels": axis.get_legend_handles_labels()[1],
            }
        elif name == "transfer_real_imag.png":
            real_axis = fig.axes[0]
            closure_line = next(
                line
                for line in real_axis.lines
                if "closure" in line.get_label()
            )
            captured[name] = {
                "closure_y": np.asarray(closure_line.get_ydata(), dtype=float),
                "legend_labels": real_axis.get_legend_handles_labels()[1],
            }

    monkeypatch.setattr(export_module, "intercept_and_export_path", capture)

    omega = np.array([1.0, 2.0, 3.0])
    w_vals = np.array([-0.8 - 1.0j, -0.5 + 0.0j, -0.2 + 1.0j])
    plot_nyquist_transfer(
        omega,
        w_vals,
        [(1.25, 2.0, 2.0)],
        {"system_id": "closure_contract", **contract},
        tmp_path,
    )

    nyquist = captured["transfer_nyquist.png"]
    np.testing.assert_allclose(
        nyquist["candidate"],
        [[expected_target, 0.0]],
    )
    assert nyquist["ylim"][0] < 0.0 < nyquist["ylim"][1]
    assert any(
        f"W={expected_target_text}" in label
        for label in nyquist["legend_labels"]
    )
    assert any(
        transfer_name in label
        for label in nyquist["legend_labels"]
    )

    components = captured["transfer_real_imag.png"]
    np.testing.assert_allclose(
        components["closure_y"],
        [expected_target, expected_target],
    )
    assert (
        f"${expected_target_text}$ closure"
        in components["legend_labels"]
    )


class _ConstantDescribingFunction:
    matrix = np.eye(1)
    input_vector = np.ones(1)
    output_vector = np.ones(1)

    @staticmethod
    def describing_function(_amplitude: float) -> float:
        return 2.0


class _ResidualMapSystem:
    parameters = {"q": 1.0}
    lure = _ConstantDescribingFunction()


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "transfer_convention",
        "harmonic_condition",
        "transfer_value",
        "expected_expression",
    ),
    [
        ("standard", "1_minus_WN", 0.5, "1-N(A)W(i\\omega)"),
        ("opposite_sign", "1_plus_WN", -0.5, "1+N(A)W(i\\omega)"),
    ],
)
def test_harmonic_residual_map_uses_declared_transfer_and_equation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transfer_convention: str,
    harmonic_condition: str,
    transfer_value: float,
    expected_expression: str,
) -> None:
    import matplotlib.axes
    import hidden_attractors.plotting.export as export_module
    import hidden_attractors.plotting.plot_df as plot_df_module

    calls: list[str] = []

    def evaluate_transfer(
        _omega,
        _q,
        _transfer_mode,
        _matrix,
        _input_vector,
        _output_vector,
        *,
        transfer_convention,
    ):
        calls.append(transfer_convention)
        return complex(transfer_value)

    captured: dict[str, object] = {}
    original_contourf = matplotlib.axes.Axes.contourf

    def capture_contourf(axis, *args, **kwargs):
        captured["log_residual"] = np.asarray(args[2], dtype=float)
        return original_contourf(axis, *args, **kwargs)

    def capture_export(fig, output_path, _figure_type):
        if Path(output_path).name == "harmonic_residual_map.png":
            captured["colorbar_label"] = fig.axes[-1].get_ylabel()

    monkeypatch.setattr(plot_df_module, "W_eval", evaluate_transfer)
    monkeypatch.setattr(matplotlib.axes.Axes, "contourf", capture_contourf)
    monkeypatch.setattr(export_module, "intercept_and_export_path", capture_export)

    plot_harmonic_residual_map(
        _ResidualMapSystem(),
        [(1.5, 1.5, 2.0)],
        {
            "system_id": "residual_contract",
            "transfer_mode": "integer",
            "transfer_convention": transfer_convention,
            "harmonic_condition": harmonic_condition,
            "amplitude_min": 1.0,
            "amplitude_max": 2.0,
            "omega_min": 1.0,
            "omega_max": 2.0,
        },
        tmp_path,
    )

    assert calls == [transfer_convention] * 150
    np.testing.assert_allclose(captured["log_residual"], -8.0)
    assert expected_expression in str(captured["colorbar_label"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("transfer_convention", "harmonic_condition"),
    [
        ("standard", "1_plus_WN"),
        ("opposite_sign", "1_minus_WN"),
    ],
)
def test_nyquist_plot_rejects_incoherent_sign_pair_without_opt_in(
    tmp_path: Path,
    transfer_convention: str,
    harmonic_condition: str,
) -> None:
    with pytest.raises(ValueError, match="Incoherent transfer/sign pair"):
        plot_nyquist_transfer(
            np.array([1.0]),
            np.array([0.5 + 0.0j]),
            [],
            {
                "system_id": "invalid_contract",
                "transfer_convention": transfer_convention,
                "harmonic_condition": harmonic_condition,
            },
            tmp_path,
        )
