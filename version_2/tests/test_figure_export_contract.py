import json
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
from hidden_attractors.plotting.export import (
    export_figure,
    promote_local_figure_pairs_batch,
    save_report_figure_pair,
)


def _write_local_pair(root, figure_id):
    root.mkdir(parents=True, exist_ok=True)
    png = root / f"{figure_id}.png"
    png.write_bytes(f"png:{figure_id}".encode())
    png.with_suffix(".pdf").write_bytes(f"pdf:{figure_id}".encode())
    return png


def _tree_snapshot(root):
    if not root.exists():
        return set(), {}
    directories = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()}
    files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    return directories, files

@pytest.mark.plotting
def test_export_figure_saves_both_formats(tmp_path, monkeypatch):
    import hidden_attractors.plotting.export as export_mod
    import hidden_attractors.plotting.manifest as manifest_mod
    
    mock_root = tmp_path / "library_figures"
    monkeypatch.setattr(export_mod, "LIBRARY_FIGURES_ROOT", mock_root)
    monkeypatch.setattr(manifest_mod, "LIBRARY_FIGURES_ROOT", mock_root)
    
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    
    test_run_id = "test_run_export_contract"
    test_fig_id = "test_contract_fig"
    metadata = {
        "caption_key": "test_fig_key",
        "source_script": "test_figure_export_contract.py",
        "source_function": "test_export_figure_saves_both_formats",
        "q": "1.0",
        "parameters": {"alpha": 10.0},
        "t_final": 10.0
    }
    
    # Export
    pdf_path, png_path = export_figure(
        fig=fig,
        figure_id=test_fig_id,
        kind="attractor",
        metadata_dict=metadata,
        run_id=test_run_id
    )
    
    plt.close(fig)
    
    # Check existence
    assert pdf_path.exists()
    assert png_path.exists()
    assert pdf_path.suffix == ".pdf"
    assert png_path.suffix == ".png"
    
    # Check metadata existence
    meta_path = mock_root / "by_run" / test_run_id / "metadata" / f"{test_fig_id}.json"
    assert meta_path.exists()
    
    # Assert paths reside within tmp_path (subpath verification)
    assert tmp_path in pdf_path.parents
    assert tmp_path in png_path.parents


@pytest.mark.plotting
def test_report_pair_preserves_panel_titles_and_export_options(tmp_path, monkeypatch):
    fig, axis = plt.subplots()
    axis.set_title("(a) finite benchmark")
    calls = []

    def record_save(path, **kwargs):
        calls.append((Path(path), kwargs))

    monkeypatch.setattr(fig, "savefig", record_save)
    pdf_path, png_path = save_report_figure_pair(
        fig,
        tmp_path / "report" / "overview",
        dpi=300,
        pad_inches=0.08,
        pdf_metadata={"Creator": "HAFO validation"},
    )

    assert axis.get_title() == "(a) finite benchmark"
    assert (pdf_path, png_path) == (
        tmp_path / "report" / "overview.pdf",
        tmp_path / "report" / "overview.png",
    )
    assert [path.suffix for path, _ in calls] == [".pdf", ".png"]
    assert calls[0][1]["metadata"] == {"Creator": "HAFO validation"}
    assert calls[0][1]["pad_inches"] == calls[1][1]["pad_inches"] == 0.08
    assert calls[1][1]["dpi"] == 300
    plt.close(fig)


@pytest.mark.plotting
def test_batch_promotion_commits_every_pair_and_manifest_together(tmp_path, monkeypatch):
    import hidden_attractors.plotting.export as export_mod
    import hidden_attractors.plotting.manifest as manifest_mod

    mock_root = tmp_path / "library_figures"
    monkeypatch.setattr(export_mod, "LIBRARY_FIGURES_ROOT", mock_root)
    monkeypatch.setattr(manifest_mod, "LIBRARY_FIGURES_ROOT", mock_root)
    first = _write_local_pair(tmp_path / "local", "first")
    second = _write_local_pair(tmp_path / "local", "second")

    results = promote_local_figure_pairs_batch(
        [
            {
                "png_path": first,
                "kind": "phase",
                "metadata_dict": {"system_id": "test", "caption_key": "first"},
                "export_targets": ["paper"],
            },
            {
                "png_path": second,
                "kind": "spectrum",
                "metadata_dict": {"system_id": "test", "caption_key": "second"},
                "export_targets": ["paper"],
            },
        ],
        run_id="one-scientific-run",
    )

    assert [pair[0].stem for pair in results] == ["first", "second"]
    for figure_id in ("first", "second"):
        assert (mock_root / "by_run" / "one-scientific-run" / "pdf" / f"{figure_id}.pdf").is_file()
        assert (mock_root / "by_run" / "one-scientific-run" / "png" / f"{figure_id}.png").is_file()
        assert (mock_root / "by_run" / "one-scientific-run" / "metadata" / f"{figure_id}.json").is_file()
        assert (mock_root / "current" / "pdf" / f"{figure_id}.pdf").is_file()
        assert (mock_root / "current" / "png" / f"{figure_id}.png").is_file()
        assert (mock_root / "by_export" / "paper" / "pdf" / f"{figure_id}.pdf").is_file()
        assert (mock_root / "by_export" / "paper" / "png" / f"{figure_id}.png").is_file()

    manifest = json.loads((mock_root / "manifests" / "figure_manifest.json").read_text(encoding="utf-8"))
    assert [entry["figure_id"] for entry in manifest] == ["first", "second"]
    assert {entry["run_id"] for entry in manifest} == {"one-scientific-run"}
    csv_text = (mock_root / "manifests" / "figure_manifest.csv").read_text(encoding="utf-8")
    assert "first" in csv_text and "second" in csv_text


@pytest.mark.plotting
@pytest.mark.parametrize("fail_at", [6, 16])
def test_batch_promotion_rolls_back_every_global_path_after_intermediate_failure(
    tmp_path, monkeypatch, fail_at
):
    import hidden_attractors.plotting.export as export_mod
    import hidden_attractors.plotting.manifest as manifest_mod

    mock_root = tmp_path / "library_figures"
    monkeypatch.setattr(export_mod, "LIBRARY_FIGURES_ROOT", mock_root)
    monkeypatch.setattr(manifest_mod, "LIBRARY_FIGURES_ROOT", mock_root)
    (mock_root / "current" / "png").mkdir(parents=True)
    (mock_root / "current" / "png" / "first.png").write_bytes(b"preexisting-current")
    (mock_root / "manifests").mkdir(parents=True)
    (mock_root / "manifests" / "figure_manifest.json").write_text("[]\n", encoding="utf-8")
    (mock_root / "manifests" / "figure_manifest.csv").write_text("sentinel-csv\n", encoding="utf-8")
    before = _tree_snapshot(mock_root)

    first = _write_local_pair(tmp_path / "local", "first")
    second = _write_local_pair(tmp_path / "local", "second")
    real_copy2 = export_mod.shutil.copy2
    global_copy_count = 0

    def fail_during_global_commit(source, destination, *args, **kwargs):
        nonlocal global_copy_count
        destination = Path(destination)
        if mock_root in destination.parents:
            global_copy_count += 1
            if global_copy_count == fail_at:
                raise OSError("injected intermediate promotion failure")
        return real_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr(export_mod.shutil, "copy2", fail_during_global_commit)
    with pytest.raises(OSError, match="injected intermediate promotion failure"):
        promote_local_figure_pairs_batch(
            [
                {
                    "png_path": first,
                    "kind": "phase",
                    "metadata_dict": {"system_id": "test"},
                    "export_targets": ["paper"],
                },
                {
                    "png_path": second,
                    "kind": "spectrum",
                    "metadata_dict": {"system_id": "test"},
                    "export_targets": ["paper"],
                },
            ],
            run_id="failed-run",
        )

    assert global_copy_count == fail_at
    assert _tree_snapshot(mock_root) == before


@pytest.mark.plotting
def test_batch_validation_fails_before_touching_global_state(tmp_path, monkeypatch):
    import hidden_attractors.plotting.export as export_mod
    import hidden_attractors.plotting.manifest as manifest_mod

    mock_root = tmp_path / "library_figures"
    monkeypatch.setattr(export_mod, "LIBRARY_FIGURES_ROOT", mock_root)
    monkeypatch.setattr(manifest_mod, "LIBRARY_FIGURES_ROOT", mock_root)
    valid = _write_local_pair(tmp_path / "local", "valid")
    incomplete = tmp_path / "local" / "incomplete.png"
    incomplete.write_bytes(b"png-only")

    with pytest.raises(FileNotFoundError):
        promote_local_figure_pairs_batch(
            [
                {"png_path": valid, "kind": "phase", "metadata_dict": {}},
                {"png_path": incomplete, "kind": "phase", "metadata_dict": {}},
            ],
            run_id="never-created",
        )

    assert not mock_root.exists()


@pytest.mark.plotting
def test_batch_post_commit_validator_failure_rolls_back_manifests_and_copies(tmp_path, monkeypatch):
    import hidden_attractors.plotting.export as export_mod
    import hidden_attractors.plotting.manifest as manifest_mod

    mock_root = tmp_path / "library_figures"
    monkeypatch.setattr(export_mod, "LIBRARY_FIGURES_ROOT", mock_root)
    monkeypatch.setattr(manifest_mod, "LIBRARY_FIGURES_ROOT", mock_root)
    (mock_root / "manifests").mkdir(parents=True)
    (mock_root / "manifests" / "figure_manifest.json").write_text("[]\n", encoding="utf-8")
    (mock_root / "manifests" / "figure_manifest.csv").write_text("old-csv\n", encoding="utf-8")
    before = _tree_snapshot(mock_root)
    local_png = _write_local_pair(tmp_path / "local", "validator_target")
    observed = {}

    def reject_after_complete_commit(**receipt):
        observed.update(receipt)
        assert all(path.is_file() for pair in receipt["promoted_pairs"] for path in pair)
        assert all(path.is_file() for path in receipt["manifest_paths"])
        raise RuntimeError("scientific source drift")

    with pytest.raises(RuntimeError, match="scientific source drift"):
        promote_local_figure_pairs_batch(
            [
                {
                    "png_path": local_png,
                    "kind": "phase",
                    "metadata_dict": {"system_id": "test"},
                    "export_targets": ["paper"],
                }
            ],
            run_id="timestamped-run-20260803T120000Z",
            validator=reject_after_complete_commit,
        )

    assert observed["run_id"] == "timestamped-run-20260803T120000Z"
    assert _tree_snapshot(mock_root) == before
