import sys
from pathlib import Path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import pytest
from src.verification.classifiers import classify_hiddenness_verdict

def test_hiddenness_verdict_language():
    # Even if target_hits_from_equilibria = 0 and stable equilibria count is 0,
    # it must return compatible_with_hiddenness_under_sampled_radii, NOT hidden_verified_under_tested_radii.
    verdict = classify_hiddenness_verdict(
        target_hits_from_equilibria=0,
        equilibria_count=3,
        unstable_equilibria_count=3,  # stable_count = 0
        seed_reached_attractor=True,
        numerical_failures=0
    )
    assert verdict == "compatible_with_hiddenness_under_sampled_radii"
    assert verdict != "hidden_verified_under_tested_radii"
