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
    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    if candidates:
        top_cand = candidates[0]
        print("Keys in top candidate record:")
        for k, v in top_cand.items():
            if k not in ["trajectory", "seed"]:
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {type(v).__name__} (len={len(v) if hasattr(v, '__len__') else 'N/A'})")

if __name__ == "__main__":
    inspect()
