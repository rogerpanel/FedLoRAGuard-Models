"""HGT-style relation-aware multi-head attention (Hu et al., 2020).

Implements Eq. (4) of the paper:

    alpha_{u,v,r} = softmax( (W_Q^{tau(v)} h_v)^T  W_R^r  (W_K^{tau(u)} h_u)
                              / sqrt(d/H) )

with type-specific Q / K / V projections, a relation-specific bilinear matrix
``W_R^r`` per edge type, and an optional Relative Temporal Encoding modulating
the attention logits.

The implementation operates over flat tensors so that it is friendly to
PyTorch Geometric's ``HeteroData`` representation as well as to the bespoke
in-memory :class:`HeteroDynamicGraph` shipped with this package.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import math

import torch
from torch import nn


class HGTRelationAwareAttention(nn.Module):
    def __init__(
        self,
        node_types: Sequence[str],
        edge_types: Sequence[str],
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_relative_temporal: bool = True,
        time_dim: int = 32,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_rte = use_relative_temporal

        self.q_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in self.node_types})
        self.k_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in self.node_types})
        self.v_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in self.node_types})
        # Relation-specific bilinear matrices.
        self.w_rel = nn.ParameterDict(
            {r: nn.Parameter(torch.eye(self.head_dim).repeat(num_heads, 1, 1)) for r in self.edge_types}
        )
        if self.use_rte:
            self.rte = nn.Linear(time_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_query: torch.Tensor,                # (Q, d)
        h_key:   torch.Tensor,                # (K, d)
        query_type: str,
        neighbor_types: Sequence[str],        # length K
        relations: Sequence[str],             # length K
        time_encoding: Optional[torch.Tensor] = None,  # (K, time_dim)
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self.q_proj[query_type](h_query)
        # Stack per-key projections by their source type.
        k_per = []
        v_per = []
        for nt, h in zip(neighbor_types, h_key):
            k_per.append(self.k_proj[nt](h))
            v_per.append(self.v_proj[nt](h))
        if not k_per:
            return self.norm(h_query)
        K = torch.stack(k_per, dim=0)
        V = torch.stack(v_per, dim=0)
        if self.use_rte and time_encoding is not None:
            K = K + self.rte(time_encoding)

        Q_h = Q.view(-1, self.num_heads, self.head_dim)             # (Q, H, d_h)
        K_h = K.view(-1, self.num_heads, self.head_dim)             # (K, H, d_h)
        V_h = V.view(-1, self.num_heads, self.head_dim)             # (K, H, d_h)

        # Apply relation-specific bilinear to K per edge.
        K_r = torch.zeros_like(K_h)
        for j, r in enumerate(relations):
            W = self.w_rel[r]                                       # (H, d_h, d_h)
            K_r[j] = torch.einsum("hij,hj->hi", W, K_h[j])

        # Attention logits: (Q, K, H)
        logits = torch.einsum("qhd,khd->qkh", Q_h, K_r) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            logits = logits.masked_fill(attn_mask.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(logits, dim=1)
        alpha = self.dropout(alpha)

        # Aggregate: (Q, H, d_h)
        out = torch.einsum("qkh,khd->qhd", alpha, V_h).reshape(-1, self.hidden_dim)
        return self.norm(h_query + self.out_proj(out))
