from __future__ import annotations

import sys
import json
import pytest
from pathlib import Path
import numpy as np
from hidden_attractors.analysis import TimeSeriesLyapunovResult
from hidden_attractors.cli.lyapunov import trajectory_lyapunov_spectrum
from hidden_attractors.cli.run import main

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))


def test_lyapunov_cli_lifecycle(tmp_path):
    config_path = (
        Path(workspace_root)
        / "version_2"
        / "tests"
        / "fixtures"
        / "software_validation_fractional.yaml"
    )
    
    # Run Lyapunov workflow via CLI with extremely short times for fast test
    main([
        "lyapunov", "compute",
        "-c", str(config_path),
        "--output_dir", str(tmp_path),
        "--lyapunov.t_final", "0.2",
        "--lyapunov.t_burn", "0.05",
        "--lyapunov.h", "0.01",
        "--use_c_backend", "false",
    ])
    
    summary_path = tmp_path / "lyapunov_summary.json"
    assert summary_path.exists()
    
    # Validate Lyapunov summary JSON via CLI command
    try:
        main([
            "lyapunov", "validate",
            "-i", str(summary_path),
        ])
    except SystemExit as e:
        assert e.code == 0
        
    # Write a mock trajectory file to test trajectory_lyapunov_spectrum command
    trajectory_csv = tmp_path / "mock_trajectory.csv"
    import csv
    with open(trajectory_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z"])
        # Needs at least 100 points
        for t_idx in range(120):
            writer.writerow([t_idx * 0.01, float(np.sin(t_idx * 0.1)), 0.0, 0.0])
            
    # Try calling trajectory-based lyapunov spectrum estimation command
    try:
        main([
            "lyapunov", "spectrum",
            "-t", str(trajectory_csv),
            "--observable", "x",
        ])
    except SystemExit as e:
        # It could exit with 1 if nolds is missing, or succeed if nolds is installed.
        # We accept either exit code or success.
        pass


def test_trajectory_lyapunov_cli_writes_structured_json(
    tmp_path,
    monkeypatch,
) -> None:
    trajectory_csv = tmp_path / "trajectory.csv"
    trajectory_csv.write_text(
        "t,x,y,z\n"
        + "".join(
            f"{index * 0.02},{np.sin(index * 0.1)},0,0\n"
            for index in range(160)
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "lyapunov.json"
    fake = TimeSeriesLyapunovResult(
        largest_exponent=0.4,
        spectrum=(0.4, 0.0, -2.0),
        kaplan_yorke_dimension=2.2,
        kaplan_yorke_status="computed_from_finite_time_eckmann_spectrum",
        spectrum_sum=-1.6,
        spectrum_status="finite_time_nolds_eckmann_scalar_reconstruction",
        sample_interval=0.02,
        sample_rate=50.0,
        time_unit="s",
        exponent_unit="s^-1",
        n_samples=160,
        estimated_pairwise_matrix_bytes=1024,
        observable="x",
        backend="nolds",
        backend_version="test",
        rosenstein_method="test-rosenstein",
        eckmann_method="test-eckmann",
        rosenstein_parameters={},
        eckmann_parameters={},
        largest_sign_agrees_with_spectrum=True,
        rosenstein_slope_per_sample=0.008,
        rosenstein_fit_r2=1.0,
        rosenstein_divergence_index_unit="retained_sample_offset",
        rosenstein_divergence_trajectory=((0.0, 0.0), (1.0, 0.008)),
        rosenstein_divergence_time_trajectory=((0.0, 0.0), (0.02, 0.008)),
        evidence_status="finite_time_time_series_diagnostic",
        warnings=("diagnostic only",),
    )
    monkeypatch.setattr(
        "hidden_attractors.cli.lyapunov.estimate_time_series_lyapunov",
        lambda *args, **kwargs: fake,
    )

    trajectory_lyapunov_spectrum(
        [
            "--trajectory",
            str(trajectory_csv),
            "--observable",
            "x",
            "--json-output",
            str(output_json),
        ]
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["analysis_type"] == "time_series_lyapunov"
    assert payload["largest_exponent"] == 0.4
    assert payload["spectrum"] == [0.4, 0.0, -2.0]
    assert payload["kaplan_yorke_dimension"] == 2.2

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "lyapunov",
                "validate",
                "--input",
                str(output_json),
            ]
        )
    assert exc_info.value.code == 0
