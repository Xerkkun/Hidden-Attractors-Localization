"""Basin-classification helpers and backend exports.

Stability: stable
    Classification labels, ``class_label``, ``is_target_class`` are fixed API.
    ``BasinBackend`` is re-exported here for convenience; see ``native`` for its
    internal contract.
"""

from .classification import CLASS_LABELS, TARGET_CLASS_IDS, class_label, is_target_class
from ..native.backends import BasinBackend

__all__ = ["BasinBackend", "CLASS_LABELS", "TARGET_CLASS_IDS", "class_label", "is_target_class"]
