import csv
import json
import math

# Load candidate scan CSV
csv_path = r"version_2/outputs/arctan_full_memory_search/run_20260605_194127/candidate_scan.csv"

target_cases = {
    "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1_rho_1p5_branch_0",
    "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1_rho_1p25_branch_0",
    "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1p2_rho_1p5_branch_0",
    "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1p2_rho_1p25_branch_0",
    "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1p5585_rho_1_branch_0",
    "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m2_rho_0p75_branch_0",
    "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m2p5_rho_0p5_branch_0",
    "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m3_rho_0p5_branch_0",
    "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m0p8_rho_2_branch_0",
    "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m1_rho_1p5_branch_0",
    "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m1p2_rho_1p25_branch_0",
    "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m1p5585_rho_1_branch_0",
    "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m2_rho_0p75_branch_0",
}

rows = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["case_id"] in target_cases:
            rows.append(row)

print(f"Found {len(rows)} of {len(target_cases)} targets in CSV.\n")

print(f"{'Case ID':<75} | {'a1':<4} | {'a2':<7} | {'rho':<5} | {'|a2|*rho':<8} | {'omega':<7} | {'k':<7} | {'A':<7}")
print("-" * 128)

parsed_seeds = []
case_ids = []

for r in rows:
    a1 = float(r["a1"])
    a2 = float(r["a2"])
    rho = float(r["rho"])
    a2_times_rho = abs(a2) * rho
    omega = float(r["omega"])
    k = float(r["k"])
    A = float(r["A"])
    case_id = r["case_id"]
    
    # parse seed
    seed_str = r["seed"]
    seed = json.loads(seed_str)
    parsed_seeds.append(seed)
    case_ids.append(case_id)
    
    print(f"{case_id:<75} | {a1:<4} | {a2:<7.4f} | {rho:<5} | {a2_times_rho:<8.4f} | {omega:<7.4f} | {k:<7.4f} | {A:<7.4f}")

# Calculate pairwise distances
print("\n" + "="*80)
print(" DISTANCE ANALYSIS BETWEEN SEEDS / INITIAL CONDITIONS")
print("="*80)

n = len(rows)
dists = []
for i in range(n):
    for j in range(i + 1, n):
        s1 = parsed_seeds[i]
        s2 = parsed_seeds[j]
        dist = math.sqrt(sum((x1 - x2)**2 for x1, x2 in zip(s1, s2)))
        dists.append((dist, case_ids[i], case_ids[j]))

dists.sort()
print(f"Number of target combinations: {len(dists)}")
print(f"Minimum distance between any two seeds: {dists[0][0]:.6f}")
print(f"  Between: {dists[0][1]}")
print(f"      and: {dists[0][2]}")
print(f"Maximum distance between any two seeds: {dists[-1][0]:.6f}")
print(f"  Between: {dists[-1][1]}")
print(f"      and: {dists[-1][2]}\n")

# Calculate average coordinates of seed
avg_x = sum(s[0] for s in parsed_seeds) / n
avg_y = sum(s[1] for s in parsed_seeds) / n
avg_z = sum(s[2] for s in parsed_seeds) / n
print(f"Average Seed Coordinate: [{avg_x:.6f}, {avg_y:.6f}, {avg_z:.6f}]")

# Calculate standard deviations of coordinates of seed
var_x = sum((s[0] - avg_x)**2 for s in parsed_seeds) / n
var_y = sum((s[1] - avg_y)**2 for s in parsed_seeds) / n
var_z = sum((s[2] - avg_z)**2 for s in parsed_seeds) / n
print(f"Std Dev of Seed Coordinates: [{math.sqrt(var_x):.6f}, {math.sqrt(var_y):.6f}, {math.sqrt(var_z):.6f}]")
