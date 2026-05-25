"""Maintained diagnostic gates for candidate trajectories."""

from .periodicity import (
    classify_post_transient_periodicity,
    promotion_label_after_hiddenness_probe,
)

__all__ = [
    "classify_post_transient_periodicity",
    "promotion_label_after_hiddenness_probe",
]
