import json
from pathlib import Path

candidate_dir = Path("outputs/arctan_hidden_candidate_search/c590_q09999_seed9_candidate_20260623")

# Load high density matrix
with open(candidate_dir / "hiddenness_matrix_scaled.json", "r", encoding="utf-8") as f:
    scaled = json.load(f)

# Load extended radii matrix
with open(candidate_dir / "hiddenness_matrix_extended_radii_12.json", "r", encoding="utf-8") as f:
    extended = json.load(f)

# Merge
merged = {}
merged["schema_version"] = "1.0"
merged["candidate_id"] = scaled["candidate_id"]
merged["status"] = "completed_merged_hiddenness_probe_matrix"
merged["parameters"] = scaled["parameters"]
merged["seed"] = scaled["seed"]
merged["q"] = scaled["q"]
merged["h"] = scaled["h"]
merged["integrator"] = scaled.get("integrator", "ABM predictor-corrector")
merged["memory_mode"] = scaled.get("memory_mode", "full")
merged["caputo_history_accumulated"] = scaled.get("caputo_history_accumulated", True)
merged["t_final"] = scaled.get("t_final", 180.0)
merged["t_burn"] = scaled.get("t_burn", 90.0)
merged["equilibria"] = scaled["equilibria"]
merged["matignon"] = scaled["matignon"]

# Combine and sort radii
all_radii = set(scaled["radii"]) | set(extended["radii"])
# Sort by numeric value
sorted_radii = sorted(all_radii, key=float)
merged["radii"] = sorted_radii

merged["target_calibration_nn90"] = scaled["target_calibration_nn90"]
merged["contact_threshold"] = scaled["contact_threshold"]

# Combine rows
rows = []
for row in scaled["rows"]:
    rows.append(row)

for row in extended["rows"]:
    rows.append(row)

merged["rows"] = rows
merged["tests"] = len(rows)
merged["contacts"] = sum(1 for row in rows if bool(row.get("contact", False)))
merged["all_equilibria_tested"] = True

# Add scientific boundary description
merged["scientific_boundary"] = scaled["scientific_boundary"]

# Write
with open(candidate_dir / "hiddenness_matrix_merged.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

print(f"Merged successfully: total rows = {len(rows)}, total contacts = {merged['contacts']}")
