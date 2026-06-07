import json
from pathlib import Path

def inspect():
    summary_path = Path(r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order\version_2\outputs\arctan_full_memory_search\run_20260605_194127\summary.json")
    if not summary_path.exists():
        print("Summary file not found.")
        return
        
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    candidates = data.get("candidates", [])
    print(f"Total candidates in this run: {len(candidates)}")
    
    # Sort candidates by score descending
    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    for i in range(min(5, len(candidates))):
        cand = candidates[i]
        print(f"\n--- Candidate Rank {i+1} ---")
        print(f"Case ID: {cand.get('case_id')}")
        print(f"Label: {cand.get('candidate_label')}")
        print(f"Score: {cand.get('score')}")
        print(f"Omega: {cand.get('omega')}, k: {cand.get('k')}, A: {cand.get('A')}")
        # Let's see if the starting seed and trajectories are present
        print(f"Seed: {cand.get('seed')}")
        print(f"Trajectory file: {cand.get('trajectory')}")
        print(f"Figures: {cand.get('figures')}")
        
if __name__ == "__main__":
    inspect()
