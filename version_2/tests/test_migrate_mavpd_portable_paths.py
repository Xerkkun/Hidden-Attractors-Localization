from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


VERSION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = VERSION_ROOT / "tools" / "release" / "migrate_mavpd_portable_paths.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("migrate_mavpd_portable_paths", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encoded(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_encoded(payload))


def _fixture(root: Path, module: ModuleType) -> dict[str, object]:
    figure_rows = []
    run_rows = []
    for figure_id in module.FIGURE_IDS:
        portable = {
            suffix: module._figure_portable_path(figure_id, suffix)
            for suffix in ("pdf", "png")
        }
        absolute = {
            suffix: str(root / "outputs" / Path(relative))
            for suffix, relative in portable.items()
        }
        figure_rows.append(
            {
                "figure_id": figure_id,
                "central_paths": absolute,
                "scientific_marker": {"unchanged": figure_id},
            }
        )
        run_rows.append(
            {
                "figure_id": figure_id,
                "central_paths": {
                    suffix: f"outputs/{relative}" for suffix, relative in portable.items()
                },
                "scientific_marker": {"unchanged": figure_id},
            }
        )
    receipt_paths = [
        "outputs/library_figures/manifests/figure_manifest.json",
        "outputs/library_figures/manifests/figure_manifest.csv",
    ]
    figure_manifest = {"figures": figure_rows, "scientific_marker": 17}
    receipt = {
        "run_id": module.RUN_ID,
        "figure_ids": list(module.FIGURE_IDS),
        "global_manifest_paths": [str(root / Path(path)) for path in receipt_paths],
        "scientific_marker": 23,
    }
    run_manifest = {
        "run_id": module.RUN_ID,
        "figures": run_rows,
        "global_figure_promotion": {
            "run_id": module.RUN_ID,
            "global_manifest_paths": receipt_paths,
        },
    }
    paths = {
        module.FIGURE_MANIFEST_RELATIVE: root / module.FIGURE_MANIFEST_RELATIVE,
        module.PROMOTION_RECEIPT_RELATIVE: root / module.PROMOTION_RECEIPT_RELATIVE,
        module.RUN_MANIFEST_RELATIVE: root / module.RUN_MANIFEST_RELATIVE,
        module.RUN_STATUS_RELATIVE: root / module.RUN_STATUS_RELATIVE,
    }
    _write(paths[module.FIGURE_MANIFEST_RELATIVE], figure_manifest)
    _write(paths[module.PROMOTION_RECEIPT_RELATIVE], receipt)
    _write(paths[module.RUN_MANIFEST_RELATIVE], run_manifest)
    status = {
        "run_id": module.RUN_ID,
        "artifacts": {
            "figures/figure_manifest.json": sha256(
                paths[module.FIGURE_MANIFEST_RELATIVE].read_bytes()
            ).hexdigest(),
            "figures/global_promotion_receipt.json": sha256(
                paths[module.PROMOTION_RECEIPT_RELATIVE].read_bytes()
            ).hexdigest(),
            "run_manifest.json": sha256(
                paths[module.RUN_MANIFEST_RELATIVE].read_bytes()
            ).hexdigest(),
        },
        "scientific_marker": {"unchanged": True},
    }
    _write(paths[module.RUN_STATUS_RELATIVE], status)
    return {
        "paths": paths,
        "figure_manifest": figure_manifest,
        "receipt": receipt,
        "run_manifest": run_manifest,
        "run_status": status,
    }


@pytest.mark.release_readiness
def test_migration_changes_exactly_36_paths_and_rebinds_only_three_hashes(tmp_path: Path) -> None:
    module = _module()
    fixture = _fixture(tmp_path, module)
    before_figures = deepcopy(fixture["figure_manifest"])
    before_receipt = deepcopy(fixture["receipt"])
    before_run_manifest = deepcopy(fixture["run_manifest"])
    before_status = deepcopy(fixture["run_status"])

    report = module.migrate(version_root=tmp_path, write=True)

    assert report["changed_path_fields"] == 36
    assert report["path_allowlist_size"] == 36
    assert report["derived_ledger_fields"] == 3
    paths = fixture["paths"]
    figures = json.loads(paths[module.FIGURE_MANIFEST_RELATIVE].read_text(encoding="utf-8"))
    receipt = json.loads(paths[module.PROMOTION_RECEIPT_RELATIVE].read_text(encoding="utf-8"))
    run_manifest = json.loads(paths[module.RUN_MANIFEST_RELATIVE].read_text(encoding="utf-8"))
    status = json.loads(paths[module.RUN_STATUS_RELATIVE].read_text(encoding="utf-8"))

    assert figures["scientific_marker"] == before_figures["scientific_marker"]
    assert receipt["scientific_marker"] == before_receipt["scientific_marker"]
    assert status["scientific_marker"] == before_status["scientific_marker"]
    assert [row["scientific_marker"] for row in figures["figures"]] == [
        row["scientific_marker"] for row in before_figures["figures"]
    ]
    assert all(
        value.startswith("library_figures/") and "\\" not in value
        for row in figures["figures"]
        for value in row["central_paths"].values()
    )
    assert receipt["global_manifest_paths"] == [
        "library_figures/manifests/figure_manifest.json",
        "library_figures/manifests/figure_manifest.csv",
    ]
    assert [row["scientific_marker"] for row in run_manifest["figures"]] == [
        row["scientific_marker"] for row in before_run_manifest["figures"]
    ]
    assert run_manifest["global_figure_promotion"]["global_manifest_paths"] == receipt[
        "global_manifest_paths"
    ]
    assert status["artifacts"]["figures/figure_manifest.json"] == sha256(
        paths[module.FIGURE_MANIFEST_RELATIVE].read_bytes()
    ).hexdigest()
    assert status["artifacts"]["figures/global_promotion_receipt.json"] == sha256(
        paths[module.PROMOTION_RECEIPT_RELATIVE].read_bytes()
    ).hexdigest()
    assert status["artifacts"]["run_manifest.json"] == sha256(
        paths[module.RUN_MANIFEST_RELATIVE].read_bytes()
    ).hexdigest()
    assert module.migrate(version_root=tmp_path, write=False)["changed_path_fields"] == 0


@pytest.mark.release_readiness
def test_migration_rejects_an_absolute_path_outside_repository(tmp_path: Path) -> None:
    module = _module()
    fixture = _fixture(tmp_path, module)
    paths = fixture["paths"]
    manifest_path = paths[module.FIGURE_MANIFEST_RELATIVE]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["figures"][0]["central_paths"]["pdf"] = str(
        tmp_path.parent / "outside" / "00_nyquist_direct_seed.pdf"
    )
    _write(manifest_path, manifest)
    status_path = paths[module.RUN_STATUS_RELATIVE]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["artifacts"]["figures/figure_manifest.json"] = sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    _write(status_path, status)
    snapshots = {path: path.read_bytes() for path in paths.values()}

    with pytest.raises(module.PathMigrationError, match="outside the repository"):
        module.migrate(version_root=tmp_path, write=True)

    assert {path: path.read_bytes() for path in paths.values()} == snapshots


@pytest.mark.release_readiness
def test_migration_rejects_a_partially_migrated_bundle(tmp_path: Path) -> None:
    module = _module()
    fixture = _fixture(tmp_path, module)
    paths = fixture["paths"]
    manifest_path = paths[module.FIGURE_MANIFEST_RELATIVE]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["figures"][0]["central_paths"]["pdf"] = module._figure_portable_path(
        module.FIGURE_IDS[0], "pdf"
    )
    _write(manifest_path, manifest)
    status_path = paths[module.RUN_STATUS_RELATIVE]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["artifacts"]["figures/figure_manifest.json"] = sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    _write(status_path, status)

    with pytest.raises(module.PathMigrationError, match="partially migrated"):
        module.migrate(version_root=tmp_path, write=True)
