"""ctypes wrappers around the C numerical backends.

Stability: internal
    These classes are consumed by workflows and solvers.  The build API
    (``FractionalChuaBackend.build()``, ``FullHistoryABMBackend.build()``,
    ``BasinBackend.build()``) is
    available to advanced users but may change as new C kernels are added.
    The ctypes signatures and compilation details are not part of the public
    surface.
"""

from .backends import BasinBackend, FractionalChuaBackend, FullHistoryABMBackend, NativeFractionalVariationalBackend
from .contracts import BackendBuildSpec, FractionalLyapunovRequest, FractionalLyapunovResult, IntegrationRequest, IntegrationResult, NativeIntegrationBackend, NativeLyapunovBackend

__all__ = [
    "BackendBuildSpec",
    "BasinBackend",
    "FractionalChuaBackend",
    "FullHistoryABMBackend",
    "FractionalLyapunovRequest",
    "FractionalLyapunovResult",
    "IntegrationRequest",
    "IntegrationResult",
    "NativeIntegrationBackend",
    "NativeLyapunovBackend",
    "NativeFractionalVariationalBackend",
]
