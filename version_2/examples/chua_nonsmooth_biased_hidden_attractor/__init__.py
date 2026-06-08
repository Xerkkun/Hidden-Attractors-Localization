#!/usr/bin/env python3
"""
__init__.py del ejemplo — Permite importar steps como módulos.
Expone las funciones principales de cada paso.
"""
from .step1_centered_reference import run_centered_reference
from .step2_biased_df_search import run_biased_df_search
from .step3_hiddenness_verification import run_hiddenness_verification
from .step4_extended_hiddenness import run_extended_hiddenness
from .step5_summarize_and_plot import run_summarize_and_plot

__all__ = [
    "run_centered_reference",
    "run_biased_df_search",
    "run_hiddenness_verification",
    "run_extended_hiddenness",
    "run_summarize_and_plot",
]
