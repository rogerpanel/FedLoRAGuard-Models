"""FedLoRAGuard: Federated DGNN with DP certificates for LoRA-adapter integrity.

This package is the reference implementation accompanying

    Anaedevha, Trofimov, Borodachev (2026), FedLoRAGuard: Federated Dynamic
    Graph Neural Networks with Differential-Privacy Certificates for
    Supply-Chain Integrity Verification of LoRA Adapter Ecosystems.

The public surface is intentionally minimal -- the entry points for
reproducing the paper are the ``scripts/`` directory and the
:mod:`fedloraguard.federated` runtime.
"""

__version__ = "0.1.0"
__all__ = [
    "graph",
    "encoders",
    "models",
    "federated",
    "privacy",
    "calibration",
    "investigation",
    "utils",
]
