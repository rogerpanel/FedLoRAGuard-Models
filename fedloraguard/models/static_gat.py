"""Static-graph GAT baseline (used in the *w/o DyGFormer* ablation row of
Table 2)."""
from __future__ import annotations

from typing import Sequence

import math

import torch
from torch import nn


class StaticGATEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.W = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.a_src = nn.ModuleList(
            [nn.Linear(self.head_dim, 1, bias=False) for _ in range(num_layers)]
        )
        self.a_dst = nn.ModuleList(
            [nn.Linear(self.head_dim, 1, bias=False) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(
        self,
        query_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_types: Sequence[str],
        relations: Sequence[str],
        rel_times: torch.Tensor,
        query_type: str = "adapter",
    ) -> torch.Tensor:
        if neighbor_features.numel() == 0:
            return self.norm(query_features)
        h = query_features.unsqueeze(0)
        nbr = neighbor_features
        for layer in range(self.num_layers):
            Wh_q = self.W[layer](h).view(-1, self.num_heads, self.head_dim)
            Wh_n = self.W[layer](nbr).view(-1, self.num_heads, self.head_dim)
            e_q = self.a_dst[layer](Wh_q)                       # (1, H, 1)
            e_n = self.a_src[layer](Wh_n)                       # (K, H, 1)
            e = torch.nn.functional.leaky_relu(e_q + e_n.transpose(0, 1), negative_slope=0.2)
            alpha = torch.softmax(e / math.sqrt(self.head_dim), dim=0)
            agg = (alpha * Wh_n).sum(dim=0).view(1, -1)
            h = self.norm(h + self.dropout(agg))
        return h.squeeze(0)
