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
            results[key] = {"total": 0, "TARGET": 0, "EQ": 0, "DIVERGED": 0, "OTHER": 0}
        
        results[key]["total"] += 1
        
        if contact:
            results[key]["TARGET"] += 1
        elif status == "converged_equilibrium_early":
            results[key]["EQ"] += 1
        elif status in {"diverged", "diverged_early", "failed", "numerical_failure"} or status != "ok":
            results[key]["DIVERGED"] += 1
        else:
            results[key]["OTHER"] += 1
            
    print(f"=== Strict Partition Analysis for {path} ===")
    for (eq, radius), counts in sorted(results.items(), key=lambda x: (x[0][0], x[0][1])):
        frac = counts['TARGET'] / counts['total']
        print(f"Eq: {eq}, r: {radius:g}, N: {counts['total']}, TARGET: {counts['TARGET']}, EQ: {counts['EQ']}, OTHER: {counts['OTHER']}, DIVERGED: {counts['DIVERGED']}, Frac. TARGET: {frac:.3f}")

if __name__ == "__main__":
    analyze_matrix("outputs/arctan_hidden_candidate_search/c590_q09999_seed9_candidate_20260623/hiddenness_matrix_scaled.json")
    analyze_matrix("outputs/arctan_hidden_candidate_search/c590_q09999_seed9_candidate_20260623/hiddenness_matrix_extended_radii_12.json")
