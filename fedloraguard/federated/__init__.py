from .client import FederatedClient
from .server import FederatedServer
from .secure_agg import SecureAggregator
from .fltrust import fltrust_score
from .runtime import run_federated_training

# The Flower runtime is optional (industrial / cross-silo deployments only).
try:
    from . import runtime_flower            # noqa: F401
    _HAS_FLOWER_MODULE = True
except Exception:                            # pragma: no cover
    _HAS_FLOWER_MODULE = False

__all__ = [
    "FederatedClient",
    "FederatedServer",
    "SecureAggregator",
    "fltrust_score",
    "run_federated_training",
    "runtime_flower",
]
