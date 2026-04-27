"""FedGraphNN with HGT aggregation baseline (He et al., 2021).

We reuse the FedLoRAGuard verifier with backbone forced to *static* HGT and
without the DyGFormer temporal patch -- this is exactly the design of
FedGraphNN's HGT cell adapted to the LoRA marketplace heterogeneity.
"""
from __future__ import annotations

from typing import Dict

from fedloraguard.models.verifier import FedLoRAGuardVerifier


class FedGraphNNHGT:
    @staticmethod
    def build(cfg: Dict) -> FedLoRAGuardVerifier:
        cfg = dict(cfg)
        cfg["model"] = dict(cfg["model"])
        cfg["model"]["backbone"] = "static_gat"      # no DyGFormer
        return FedLoRAGuardVerifier(
            feature_dims=cfg["graph"]["feature_dims"],
            node_types=cfg["graph"]["node_types"],
            edge_types=cfg["graph"]["edge_types"],
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"]["num_layers"],
            num_heads=cfg["model"]["num_heads"],
            dropout=cfg["model"]["dropout"],
            backbone="static_gat",
            patch_size=cfg["graph"]["temporal"]["patch_size"],
            time_dim=cfg["graph"]["temporal"]["time_encoding_dim"],
            use_relative_temporal=False,
        )
