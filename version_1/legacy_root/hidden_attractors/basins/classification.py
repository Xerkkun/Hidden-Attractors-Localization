"""Operational basin-classification labels used by the C backend.

The labels are deliberately numerical/operational.  A ``target_positive`` or
``target_negative`` hit means the classifier assigned the trajectory to the
bounded nontrivial target class under the tested numerical contract; it is not
by itself a hidden-attractor proof.
"""

from __future__ import annotations


CLASS_LABELS = {
    0: "equilibrium",
    1: "target_positive",
    2: "target_negative",
    3: "infinity",
    4: "unknown",
    5: "numerical_failure",
}
TARGET_CLASS_IDS = frozenset({1, 2})


def class_label(class_id: int) -> str:
    """Return the stable label for a basin-classification integer."""

    return CLASS_LABELS.get(int(class_id), f"class_{int(class_id)}")


def is_target_class(class_id: int) -> bool:
    """Return whether a classifier ID is one of the target-attractor classes."""

    return int(class_id) in TARGET_CLASS_IDS
