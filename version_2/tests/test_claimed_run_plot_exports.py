"""A fresh scientific run cannot implicitly replace the central figure bank."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pytest

from hidden_attractors._run_output import claim_output_directory, complete_output_claim
from hidden_attractors.plotting import export as exports
from hidden_attractors.plotting import manifest as manifests


@pytest.fixture
def bank(tmp_path, monkeypatch):
    root = tmp_path/"isolated_library_figures"
    monkeypatch.setattr(exports, "LIBRARY_FIGURES_ROOT", root)
    monkeypatch.setattr(manifests, "LIBRARY_FIGURES_ROOT", root)
    for relative in ("current/png/same.png", "by_run/default_run/png/same.png"):
        path = root/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b"recorded scientific figure")
    return root


def snapshot(root):
    return {path.relative_to(root).as_posix():path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def figure():
    fig, axis = plt.subplots(figsize=(3,2))
    axis.plot([0,1],[1,0])
    axis.set_title("removed by the paired export policy")
    return fig, axis


def test_two_claimed_roots_with_same_figure_id_leave_entire_bank_unchanged(tmp_path, monkeypatch, bank):
    before = snapshot(bank)
    for name in ("run_one", "run_two"):
        claim = claim_output_directory(tmp_path/name, run_kind="test")
        target = claim.path/"figures/same.png"
        fig, axis = figure()
        try:
            returned = exports.intercept_and_export_path(fig,target,"basin")
            assert returned == str(target)
            assert target.read_bytes().startswith(b"\x89PNG")
            assert target.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
            assert axis.get_title() == ""
        finally:
            plt.close(fig)
        complete_output_claim(claim)
    assert snapshot(bank) == before


def test_completed_claim_rejects_even_new_figure_name_before_any_write(tmp_path, monkeypatch, bank):
    claim = claim_output_directory(tmp_path/"sealed",run_kind="test")
    complete_output_claim(claim)
    before, marker = snapshot(bank), (claim.path/".hafo_run.json").read_bytes()
    fig,_ = figure()
    try:
        with pytest.raises(FileExistsError,match="completed"):
            exports.intercept_and_export_path(fig,claim.path/"new_figures/unseen.png","basin")
    finally:
        plt.close(fig)
    assert not (claim.path/"new_figures").exists()
    assert (claim.path/".hafo_run.json").read_bytes() == marker
    assert snapshot(bank) == before


@pytest.mark.parametrize("existing_suffix",[".png",".pdf"])
def test_active_claim_rejects_existing_pair_member_without_replacement(tmp_path, monkeypatch, bank,existing_suffix):
    claim = claim_output_directory(tmp_path/"active",run_kind="test")
    target = claim.path/"same.png"
    existing = target.with_suffix(existing_suffix)
    existing.write_bytes(b"keep original bytes")
    before = snapshot(bank)
    fig,_ = figure()
    try:
        with pytest.raises(FileExistsError,match="already exists"):
            exports.intercept_and_export_path(fig,target,"basin")
    finally:
        plt.close(fig)
    assert existing.read_bytes() == b"keep original bytes"
    assert not target.with_suffix(".pdf" if existing_suffix == ".png" else ".png").exists()
    assert snapshot(bank) == before


@pytest.mark.parametrize("marker",["not JSON","[]",'{"status":"in_progress"}'])
def test_invalid_marker_never_falls_back_to_global_promotion(tmp_path,monkeypatch,bank,marker):
    root=tmp_path/"invalid"
    root.mkdir()
    (root/".hafo_run.json").write_text(marker,encoding="utf-8")
    before=snapshot(bank)
    fig,_=figure()
    try:
        with pytest.raises(ValueError,match="Invalid enclosing run marker"):
            exports.intercept_and_export_path(fig,root/"same.png","basin")
    finally:
        plt.close(fig)
    assert set(path.name for path in root.iterdir()) == {".hafo_run.json"}
    assert snapshot(bank) == before


def test_explicit_global_api_and_unclaimed_legacy_interception_still_work(tmp_path,monkeypatch,bank):
    fig,_=figure()
    try:
        pdf,png=exports.export_figure(fig,"explicitly_promoted","basin",{},run_id="explicit_run")
        assert pdf.is_file() and png.is_file()
        assert (bank/"current/png/explicitly_promoted.png").is_file()
        output=tmp_path/"unclaimed/legacy.png"
        exports.intercept_and_export_path(fig,output,"basin")
        assert output.is_file() and output.with_suffix(".pdf").is_file()
        assert (bank/"current/png/legacy.png").is_file()
        assert any((bank/"by_run").rglob("legacy.png"))
    finally:
        plt.close(fig)


def test_active_claim_requires_separate_explicit_promotion_for_targets(tmp_path,monkeypatch,bank):
    claim=claim_output_directory(tmp_path/"active",run_kind="test")
    before=snapshot(bank)
    fig,_=figure()
    try:
        with pytest.raises(ValueError,match="explicit promotion"):
            exports.intercept_and_export_path(fig,claim.path/"same.png","basin",export_targets=["paper"])
    finally:
        plt.close(fig)
    assert set(path.name for path in claim.path.iterdir()) == {".hafo_run.json"}
    assert snapshot(bank) == before


def test_new_claimed_child_inside_completed_parent_is_a_distinct_run(tmp_path,monkeypatch,bank):
    parent=claim_output_directory(tmp_path/"previous",run_kind="test")
    complete_output_claim(parent)
    previous_marker=(parent.path/".hafo_run.json").read_bytes()
    child=claim_output_directory(parent.path/"new_child",run_kind="test")
    before=snapshot(bank)
    fig,_=figure()
    try:
        exports.intercept_and_export_path(fig,child.path/"same.png","basin")
    finally:
        plt.close(fig)
    assert (child.path/"same.png").is_file()
    assert (parent.path/".hafo_run.json").read_bytes() == previous_marker
    assert snapshot(bank) == before


def test_symlink_inside_claim_cannot_escape_ownership(tmp_path,monkeypatch,bank):
    claim=claim_output_directory(tmp_path/"active",run_kind="test")
    outside=tmp_path/"outside"
    outside.mkdir()
    try:
        (claim.path/"linked").symlink_to(outside,target_is_directory=True)
    except OSError:
        pytest.skip("This Windows account cannot create directory symlinks")
    before=snapshot(bank)
    fig,_=figure()
    try:
        with pytest.raises(ValueError,match="outside"):
            exports.intercept_and_export_path(fig,claim.path/"linked/same.png","basin")
    finally:
        plt.close(fig)
    assert not list(outside.iterdir())
    assert snapshot(bank) == before
