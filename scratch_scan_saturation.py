import sys
from pathlib import Path
import numpy as np

# Ensure version_2 is in python path
sys.path.insert(0, str(Path(r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order\version_2")))

from hidden_attractors.models.chua import ChuaParameters
from hidden_attractors.seed_generation.chua import find_omega_gain_candidates

def main():
    params = ChuaParameters(
        model="nonsmooth",
        alpha=8.4562,
        beta=12.0732,
        gamma=0.0052,
        m0=-0.1768,
        m1=-1.1468
    )
    
    print("Scanning for default q = 0.9998...")
    try:
        candidates = find_omega_gain_candidates(q=0.9998, params=params)
        print(f"Candidates for q=0.9998: {candidates}")
    except Exception as e:
        print(f"Error for q=0.9998: {e}")
        
    print("\nScanning for q = 0.99...")
    try:
        candidates = find_omega_gain_candidates(q=0.99, params=params)
        print(f"Candidates for q=0.99: {candidates}")
    except Exception as e:
        print(f"Error for q=0.99: {e}")
        
    print("\nScanning for q = 0.95...")
    try:
        candidates = find_omega_gain_candidates(q=0.95, params=params)
        print(f"Candidates for q=0.95: {candidates}")
    except Exception as e:
        print(f"Error for q=0.95: {e}")

if __name__ == "__main__":
    main()
