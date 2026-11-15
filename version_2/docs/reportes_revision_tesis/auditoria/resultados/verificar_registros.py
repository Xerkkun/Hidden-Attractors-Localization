"""Recount preserved records and hashes. Does not integrate or modify sources."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from math import dist


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    codes = Path("C:/Users/moren/Desktop/Codes")
    parser.add_argument("--hafo-root", type=Path, default=codes / "Hidden Attractors Fractional Order")
    parser.add_argument("--julia-root", type=Path, default=codes / "julia-dynamical-systems-demo")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    repo, v, julia = args.hafo_root, args.hafo_root / "version_2", args.julia_root / "output"
    sources: dict[str, dict] = {}

    def record(path: Path) -> Path:
        sources[str(path)] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        return path

    counts = []
    campaigns = {}
    c01_path = record(v / "validation/reference_cases/chua_integer_q1/06_hiddenness/hiddenness_validation_summary.json")
    c01 = load_json(c01_path)["checks"]
    radius_rows = load_json(record(c01_path.parent / "hidden_target_summary_integer.json"))["summary_by_radius"]
    c01_counts = {label: sum(int(row[label]) for row in radius_rows) for label in c01["global_counts"]}
    if c01_counts != c01["global_counts"] or sum(c01_counts.values()) != c01["detail_rows"]:
        raise ValueError("C01 radius summaries disagree with the stored global counts")
    campaigns["C01"] = {"kind": "recount_of_stored_radius_summaries", "summary_rows": len(radius_rows),
                         "rows_declared": c01["detail_rows"], "counts": c01_counts}
    for label, n in c01_counts.items():
        counts.append({"case": "C01", "equilibrium": "all", "radius": "all", "outcome": label, "count": n})

    c02_path = record(v / "validation/paper07_chua/evidence/nonsmooth_corrected/extended_first_contact_clean/extended_probe_runs.csv")
    c02 = load_rows(c02_path)
    if len({r["probe_id"] for r in c02}) != len(c02):
        raise ValueError("Duplicate PWL probe_id")
    local = [r for r in c02 if float(r["radius"]) <= 0.010000001]
    campaigns["C02"] = {"rows": len(c02), "unique_probe_ids": len(c02), "counts": dict(Counter(r["destination"] for r in c02)),
                         "local_rows": len(local), "local_counts": dict(Counter(r["destination"] for r in local))}
    for (eq, radius, label), n in sorted(Counter((r["equilibrium"], float(r["radius"]), r["destination"]) for r in c02).items()):
        counts.append({"case": "C02", "equilibrium": eq, "radius": radius, "outcome": label, "count": n})

    c03 = []
    for path in sorted((v / "validation/chua_fractional_arctan_c590").glob("hiddenness*_rows.csv")):
        c03.extend(load_rows(record(path)))
    keys = {(r["equilibrium"], float(r["radius"]), r["direction_index"]) for r in c03}
    if len(keys) != len(c03):
        raise ValueError("Duplicate c590 composite key")
    local = [r for r in c03 if float(r["radius"]) <= 0.300000001]
    campaigns["C03"] = {"rows": len(c03), "unique_composite_keys": len(keys), "counts": dict(Counter(r["outcome"] for r in c03)),
                         "local_rows": len(local), "local_hits": sum(r["contact"].lower() == "true" for r in local)}
    for (eq, radius, label), n in sorted(Counter((r["equilibrium"], float(r["radius"]), r["outcome"]) for r in c03).items()):
        counts.append({"case": "C03", "equilibrium": eq, "radius": radius, "outcome": label, "count": n})

    c05_path = record(v / "validation/reference_cases/kalman_fitts_integer_q1/05_verification_summary.json")
    c05 = load_json(c05_path)["hiddenness"]["probes"]
    campaigns["C05"] = {"rows": len(c05), "counts": dict(Counter(r["final_class"] for r in c05)), "hits": sum(r["target_hit"] for r in c05)}
    for label, n in campaigns["C05"]["counts"].items():
        counts.append({"case": "C05", "equilibrium": "E0", "radius": "all", "outcome": label, "count": n})

    pairs, comparisons = [], {}
    for case, python_rel, julia_rel, key_fields, fields in [
        ("C04", "pll_lead_lag_integer_q1_full/05_hiddenness_probes.csv", "pll_lead_lag_integer/full/hiddenness_probes.csv",
         ["sample_id"], ["equilibrium", "status", "final_class", "target_hit"]),
        ("C06", "mavpd_integer_hidden_chaos/07_hiddenness_probes.csv", "mavpd_integer_hidden_chaos/full/hiddenness_probes.csv",
         ["contract", "sample_id"], ["equilibrium", "status", "destination", "target_hit", "ambiguous"]),
    ]:
        a = load_rows(record(v / "validation/reference_cases" / python_rel))
        b = load_rows(record(julia / julia_rel))
        aa = {tuple(r[k] for k in key_fields): r for r in a}
        bb = {tuple(r[k] for k in key_fields): r for r in b}
        if len(aa) != len(a) or len(bb) != len(b) or aa.keys() != bb.keys():
            raise ValueError(f"Invalid pairing for {case}")
        for key, x in aa.items():
            y = bb[key]
            matches = all(x[k].lower() == y[k].lower() for k in fields)
            xa = [float(x[k]) for k in ["u0", "v0"]] if case == "C04" else json.loads(x["x0"])
            ya = [float(y[k]) for k in ["u0", "v0"]] if case == "C04" else json.loads(y["x0"])
            pairs.append({"case": case, "key": "|".join(key), "initial_distance": dist(xa, ya), "decisions_match": matches,
                          "python_destination": x.get("final_class", x.get("destination")), "julia_destination": y.get("final_class", y.get("destination"))})
        label_field = "final_class" if case == "C04" else "destination"
        campaigns[case] = {"rows": len(a), "counts": dict(Counter(r[label_field] for r in a)), "hits": sum(r["target_hit"].lower() == "true" for r in a)}
        for label, n in campaigns[case]["counts"].items():
            counts.append({"case": case, "equilibrium": "all", "radius": "all", "outcome": label, "count": n})
        if case == "C04":
            manifest = load_json(record(julia / "comparison/pll_lead_lag_integer_comparison_full.json"))
            hashes = manifest["integrity_seals"]["julia_main"]["verified_artifacts"]
            base = julia / "pll_lead_lag_integer/full"
        else:
            manifest = load_json(record(julia / "mavpd_integer_hidden_chaos/full/run_manifest.json"))
            hashes = manifest["scientific_output_hashes"]
            base = julia / "mavpd_integer_hidden_chaos/full"
        errors = [name for name, h in hashes.items() if not (base / name).exists() or sha(base / name).lower() != h.lower()]
        selected = [r for r in pairs if r["case"] == case]
        comparisons[case] = {"pairs": len(selected), "decisions_match": sum(r["decisions_match"] for r in selected),
                             "max_initial_distance": max(r["initial_distance"] for r in selected), "julia_artifacts": len(hashes), "hash_errors": errors}

    manifest_path = record(v / "docs/master_report_geometric_topological/figures/paper07/paper07_homogeneous_figures_manifest.json")
    manifest = load_json(manifest_path)
    missing, changed = [], []
    for item in manifest["inputs"]:
        path = repo / item["path"]
        if not path.exists():
            missing.append(item["path"])
        elif sha(record(path)).lower() != item["sha256"].lower():
            changed.append(item["path"])
    output_errors = [r["path"] for r in manifest["outputs"] if not (manifest_path.parent / r["path"]).exists()
                     or sha(manifest_path.parent / r["path"]).lower() != r["sha256"].lower()]

    pilot_path = v / "validation/06_geometric_topological_campaign/runs/tg_pll_mavpd_wu_b0_b2_20260812"
    pilot_rows = load_rows(record(pilot_path / "trajectory_metrics.csv"))
    pilot = {"seeds": len(load_rows(record(pilot_path / "seed_bank.csv"))), "trajectory_records": len(pilot_rows),
             "trajectory_files": len(list((pilot_path / "trajectories").glob("*.csv"))),
             "edge_records": len(load_rows(record(pilot_path / "edge_brackets.csv"))),
             "classifier_ids": dict(Counter(r["classifier_id"] for r in pilot_rows)),
             "integration_status": dict(Counter(r["integration_status"] for r in pilot_rows)),
             "destinations": dict(Counter(r["destination_label"] for r in pilot_rows))}
    report = {"scope": "Recounts, pairing of archived finite records and artifact hashes; no integration, new Lyapunov estimate, chaos proof or global hiddenness proof.",
              "campaigns": campaigns, "archived_cross_runtime_comparisons": comparisons,
              "paper07": {"inputs": len(manifest["inputs"]), "present": len(manifest["inputs"]) - len(missing), "missing": missing,
                          "input_hash_mismatches": changed, "output_figures": len(manifest["outputs"]), "output_hash_mismatches": output_errors},
              "pilot": pilot, "sources": sources}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "verificacion_resultados.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    for name, values in [("conteos_campanas.csv", counts), ("cotejo_pll_mavpd.csv", pairs)]:
        with (args.output / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    print(json.dumps({"campaigns": {k: r.get("rows", r.get("rows_declared")) for k, r in campaigns.items()},
                      "comparisons": comparisons, "paper07_present": report["paper07"]["present"], "pilot": pilot}, ensure_ascii=True))
    if changed or output_errors or any(r["hash_errors"] or r["decisions_match"] != r["pairs"] for r in comparisons.values()):
        raise SystemExit("Mismatch in preserved artifacts")


if __name__ == "__main__":
    main()
