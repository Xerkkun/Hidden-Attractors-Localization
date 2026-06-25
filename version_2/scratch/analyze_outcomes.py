import json

def analyze_matrix(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = {}
    for row in data.get("rows", []):
        eq = row.get("equilibrium")
        radius = float(row.get("radius"))
        contact = bool(row.get("contact", False))
        status = row.get("status")
        
        key = (eq, radius)
        if key not in results:
            results[key] = {"total": 0, "contacts": 0, "eq_conv": 0, "diverged": 0, "ok": 0}
        
        results[key]["total"] += 1
        if contact:
            results[key]["contacts"] += 1
        
        if status == "ok":
            results[key]["ok"] += 1
        elif status == "converged_equilibrium_early":
            results[key]["eq_conv"] += 1
        elif status in {"diverged", "diverged_early"}:
            results[key]["diverged"] += 1
        else:
            results[key]["diverged"] += 1 # Lump numerical failures / other issues in diverged or similar
            
    print(f"=== Analysis for {path} ===")
    for (eq, radius), counts in sorted(results.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"Eq: {eq}, r: {radius:g}, N: {counts['total']}, TARGET: {counts['contacts']}, EQ: {counts['eq_conv']}, OTHER: {counts['ok']}, DIVERGED: {counts['diverged']}")

if __name__ == "__main__":
    analyze_matrix("outputs/arctan_hidden_candidate_search/c590_q09999_seed9_candidate_20260623/hiddenness_matrix_scaled.json")
    analyze_matrix("outputs/arctan_hidden_candidate_search/c590_q09999_seed9_candidate_20260623/hiddenness_matrix_extended_radii_12.json")
