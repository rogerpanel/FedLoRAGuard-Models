"""Top-level FedLoRAGuard verifier: multimodal encoder -> dynamic GNN ->
classifier head, with optional temperature-scaled probability output."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from ..encoders.multimodal import MultimodalEncoder
from .dygformer import DyGFormerEncoder
from .dyg_mamba import DyGMambaEncoder
from .static_gat import StaticGATEncoder


def _split_modality(x: torch.Tensor, dims: Dict[str, int]) -> Dict[str, torch.Tensor]:
    s = 0
    out: Dict[str, torch.Tensor] = {}
    for k in ("weight", "text", "behavioral"):
        d = dims[k]
        out[k] = x[..., s:s + d]
        s += d
    return out


class FedLoRAGuardVerifier(nn.Module):
    def __init__(
        self,
        feature_dims: Dict[str, int],
        node_types: Sequence[str],
        edge_types: Sequence[str],
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        backbone: str = "dygformer",
        patch_size: int = 4,
        time_dim: int = 32,
        use_relative_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dims = feature_dims
        self.encoder = MultimodalEncoder(
            weight_dim=feature_dims["weight"],
            text_dim=feature_dims["text"],
            behavioral_dim=feature_dims["behavioral"],
            fused_dim=feature_dims.get("fused", hidden_dim),
            num_heads=num_heads,
            dropout=dropout,
        )
        self.proj_in = nn.Linear(feature_dims.get("fused", hidden_dim), hidden_dim)
        if backbone == "dygformer":
            self.gnn = DyGFormerEncoder(
                node_types=node_types,
                edge_types=edge_types,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                patch_size=patch_size,
                time_dim=time_dim,
                use_relative_temporal=use_relative_temporal,
            )
        elif backbone == "dyg_mamba":
            self.gnn = DyGMambaEncoder(hidden_dim=hidden_dim, num_layers=num_layers)
        elif backbone == "static_gat":
            self.gnn = StaticGATEncoder(
                hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout
            )
        else:
            raise ValueError(f"unknown backbone {backbone}")

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.log_temperature = nn.Parameter(torch.zeros(1))

    # ---- node-feature path -------------------------------------------------
    def encode_features(self, raw_feat: torch.Tensor) -> torch.Tensor:
        parts = _split_modality(raw_feat, self.feature_dims)
        z = self.encoder(parts["weight"], parts["text"], parts["behavioral"])
        return self.proj_in(z)

    def forward_one(
        self,
        query_feat: torch.Tensor,                 # (d_in,)
        neighbor_feats: torch.Tensor,             # (K, d_in)
        neighbor_types: Sequence[str],
        relations: Sequence[str],
        rel_times: torch.Tensor,
        query_type: str = "adapter",
    ) -> torch.Tensor:
        q = self.encode_features(query_feat.unsqueeze(0)).squeeze(0)
        if neighbor_feats.numel() == 0:
            h = q
        else:
            n = self.encode_features(neighbor_feats)
            h = self.gnn.encode(q, n, neighbor_types, relations, rel_times, query_type=query_type)
        return self.classifier(h)

    def forward_batch(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        logits = []
        for b in batch:
            logits.append(self.forward_one(
                query_feat=b["query_feat"],
                neighbor_feats=b["neighbor_feats"],
                neighbor_types=b["neighbor_types"],
                relations=b["relations"],
                rel_times=b["rel_times"],
                query_type=b.get("query_type", "adapter"),
            ))
        return torch.stack(logits, dim=0)

    def predict_proba(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward_batch(batch)
            T = torch.exp(self.log_temperature).clamp(min=1e-3)
            probs = torch.softmax(logits / T, dim=-1)
        return probs


def build_verifier(cfg: Dict[str, Any]) -> FedLoRAGuardVerifier:
    g = cfg["graph"]
    m = cfg["model"]
    return FedLoRAGuardVerifier(
        feature_dims=g["feature_dims"],
        node_types=g["node_types"],
        edge_types=g["edge_types"],
        hidden_dim=m["hidden_dim"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        dropout=m["dropout"],
        backbone=m.get("backbone", "dygformer"),
        patch_size=g["temporal"]["patch_size"],
        time_dim=g["temporal"]["time_encoding_dim"],
        use_relative_temporal=m.get("hgt", {}).get("use_relative_temporal", True),
    )
