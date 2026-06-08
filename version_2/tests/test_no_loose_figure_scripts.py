# -*- coding: utf-8 -*-
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]

PROHIBITED_ACTIVE_PATHS = [
    "scratch/replot_biased_lyapunov.py",
    "scratch/compute_biased_lyapunov_two_methods.py",
    "scratch/generate_biased_candidate_report_assets.py",
    "scratch_plot_saturation_attractor.py",
    "generate_all_plots_and_summary.py",
    "plot_chaotic_candidates.py",
    "search_saturation_candidates.py",
    "search_arctan_fractional.py",
    "compare_solvers_saturation.py",
    "version_2/run_hiddenness_biased_candidates.py",
    "version_2/search_saturation_biased_candidates.py",
    "version_2/search_saturation_biased_candidates_corrected.py",
    "version_2/examples/chua_nonsmooth_biased_hidden_attractor/step1_centered_reference.py",
    "version_2/examples/chua_nonsmooth_biased_hidden_attractor/step2_biased_df_search.py",
    "version_2/examples/chua_nonsmooth_biased_hidden_attractor/step3_hiddenness_verification.py",
    "version_2/examples/chua_nonsmooth_biased_hidden_attractor/step4_extended_hiddenness.py",
    "version_2/examples/chua_nonsmooth_biased_hidden_attractor/step5_summarize_and_plot.py",
    "version_2/tools/legacy/biased_describing_function.py",
    "version_2/tools/legacy/lure_biased_multiparam_search.py",
    "version_2/tools/legacy/lure_biased_multiparam_continuation.py",
]

def test_no_loose_or_duplicate_scripts():
    """Verify that none of the prohibited scripts exist in their original active paths."""
    for rel_path in PROHIBITED_ACTIVE_PATHS:
        full_path = ROOT_DIR / rel_path
        assert not full_path.exists(), f"El script obsoleto no debe existir en su ruta activa original: {rel_path}"
