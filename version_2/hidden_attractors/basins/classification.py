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
    6: "equilibrium_visit",
}
TERMINAL_CLASS_LABELS = {
    **CLASS_LABELS,
    0: "equilibrium_terminal",
    1: "bounded_positive_mean",
    2: "bounded_negative_mean",
    3: "threshold_crossed",
    4: "inconclusive",
}
TARGET_CLASS_IDS = frozenset({1, 2})


def class_label(class_id: int, *, classifier_policy: str = "legacy_v1") -> str:
    """Label a finite class; preserve legacy text unless a policy is supplied."""

    if classifier_policy not in {"legacy_v1", "terminal_v2"}:
        raise ValueError("classifier_policy must be 'terminal_v2' or 'legacy_v1'.")
    labels = TERMINAL_CLASS_LABELS if classifier_policy == "terminal_v2" else CLASS_LABELS
    return labels.get(int(class_id), f"class_{int(class_id)}")


def is_target_class(class_id: int, *, classifier_policy: str = "legacy_v1") -> bool:
    """Interpret old target IDs without upgrading terminal mean-sign evidence.

    The legacy default preserves historical readers. Under ``terminal_v2``,
    IDs 1/2 encode bounded responses separated by mean sign; that policy has
    no reference-contact test, so neither ID establishes a target match.
    Consumers of ``BasinBackend`` must pass its classifier policy explicitly.
    """
    if classifier_policy not in {"legacy_v1", "terminal_v2"}:
        raise ValueError("classifier_policy must be 'terminal_v2' or 'legacy_v1'.")
    return classifier_policy == "legacy_v1" and int(class_id) in TARGET_CLASS_IDS
