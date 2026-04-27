"""Reference baseline implementations used by Section 6 of the paper.

Each baseline exposes a ``train(...)`` and ``predict(...)`` function with the
same signature as the FedLoRAGuard verifier so that ``scripts/evaluate.py``
can drive every baseline through one shared evaluation loop.
"""
from .peftguard import PEFTGuardCentralized
from .shadowgenes import ShadowGenesCentralized
from .fedavg_mlp import FedAvgMLP
from .dp_fedavg_mlp import DPFedAvgMLP
from .fedgraphnn_hgt import FedGraphNNHGT
from .krum_dygformer import KrumDyGFormer

__all__ = [
    "PEFTGuardCentralized",
    "ShadowGenesCentralized",
    "FedAvgMLP",
    "DPFedAvgMLP",
    "FedGraphNNHGT",
    "KrumDyGFormer",
]
