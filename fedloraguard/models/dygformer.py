"""DyGFormer encoder (Yu et al., 2023, NeurIPS) adapted for the
heterogeneous LoRA-ecosystem CT-DG.

For each query node the encoder samples the most recent ``K`` temporal
neighbors across all relations, builds neighbor co-occurrence patches of
size ``patch_size``, and applies a Transformer encoder with relative
temporal encodings.  The aggregated representation is fed into the HGT
relation-aware attention block (:class:`HGTRelationAwareAttention`) to
mix in type-specific information.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from ..graph.temporal import BochnerTimeEncoding
from .hgt_attention import HGTRelationAwareAttention


class _NeighborCoOccurrenceEncoder(nn.Module):
    """Tally how often a neighbor type appears in the recent history (paper
    Section 4.2; DyGFormer's neighbor co-occurrence channel)."""

    def __init__(self, num_types: int, dim: int) -> None:
        super().__init__()
        self.num_types = num_types
        self.proj = nn.Linear(num_types, dim)

    def forward(self, neighbor_type_ids: torch.Tensor) -> torch.Tensor:
        if neighbor_type_ids.numel() == 0:
            return torch.zeros(0, self.proj.out_features, device=self.proj.weight.device)
        counts = torch.zeros(self.num_types, device=neighbor_type_ids.device)
        for tid in neighbor_type_ids.tolist():
            counts[int(tid)] += 1.0
        feat = counts / counts.sum().clamp(min=1.0)
        return self.proj(feat).unsqueeze(0).expand(neighbor_type_ids.shape[0], -1)


class DyGFormerEncoder(nn.Module):
    def __init__(
        self,
        node_types: Sequence[str],
        edge_types: Sequence[str],
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        patch_size: int = 4,
        time_dim: int = 32,
        use_relative_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.patch_size = max(1, int(patch_size))
        self.time_dim = time_dim

        self.time_enc = BochnerTimeEncoding(dim=time_dim, learnable=True)
        self.cooc = _NeighborCoOccurrenceEncoder(num_types=len(self.node_types), dim=hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hgt_blocks = nn.ModuleList(
            [
                HGTRelationAwareAttention(
                    node_types=self.node_types,
                    edge_types=self.edge_types,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_relative_temporal=use_relative_temporal,
                    time_dim=time_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def encode(
        self,
        query_features: torch.Tensor,                    # (Q, d)
        neighbor_features: torch.Tensor,                 # (K, d) flat per query
        neighbor_types: Sequence[str],
        relations: Sequence[str],
        rel_times: torch.Tensor,                         # (K,)
        query_type: str = "adapter",
    ) -> torch.Tensor:
        """Single-query path; called repeatedly by the verifier."""
        if neighbor_features.numel() == 0:
            return query_features

        time_codes = self.time_enc(rel_times)            # (K, time_dim)

        # Patch the K neighbors into floor(K / patch_size) patches.
        K = neighbor_features.shape[0]
        n_patches = max(1, K // self.patch_size)
        usable = n_patches * self.patch_size
        nbr = neighbor_features[:usable]
        nbr = nbr.view(n_patches, self.patch_size, -1).mean(dim=1)        # (P, d)
        # Concatenate query token at position 0 to enable [CLS]-style readout.
        # query_features: (d,) -> (1, 1, d);  nbr: (P, d) -> (1, P, d)
        seq = torch.cat(
            [query_features.unsqueeze(0).unsqueeze(0), nbr.unsqueeze(0)], dim=1
        )
        seq = self.transformer(seq).squeeze(0)
        cls = seq[0]

        # Then: HGT relation-aware mixing on the original (non-patched) neighbors.
        nbr_type_ids = torch.tensor(
            [self.node_types.index(t) for t in neighbor_types],
            dtype=torch.long, device=query_features.device,
        )
        cooc = self.cooc(nbr_type_ids)
        h_neighbors = neighbor_features + cooc
        h = cls.unsqueeze(0)
        for block in self.hgt_blocks:
            h = block(
                h_query=h, h_key=h_neighbors,
                query_type=query_type,
                neighbor_types=neighbor_types,
                relations=relations,
                time_encoding=time_codes,
            )
        return self.dropout(h.squeeze(0))
