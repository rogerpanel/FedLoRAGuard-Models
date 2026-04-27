"""Multimodal node-feature encoder (Section 4.1, Eq. (3)).

Three streams (weight, text, behavioral) are projected to a common dimension
``d_v`` and fused either by cross-attention or by simple linear projection
following our prior multimodal architecture (Anaedevha & Trofimov, 2025).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class MultimodalEncoder(nn.Module):
    def __init__(
        self,
        weight_dim: int,
        text_dim: int,
        behavioral_dim: int,
        fused_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        fusion: str = "cross_attention",
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.proj_w = nn.Linear(weight_dim, fused_dim)
        self.proj_t = nn.Linear(text_dim, fused_dim)
        self.proj_b = nn.Linear(behavioral_dim, fused_dim)
        self.dropout = nn.Dropout(dropout)
        if fusion == "cross_attention":
            self.attn = nn.MultiheadAttention(
                fused_dim, num_heads=num_heads, batch_first=True, dropout=dropout
            )
            self.ffn = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, fused_dim * 2),
                nn.GELU(),
                nn.Linear(fused_dim * 2, fused_dim),
            )
        else:
            self.fuse = nn.Linear(3 * fused_dim, fused_dim)
        self.out_norm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        w: torch.Tensor,
        t: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        # w/t/b are (B, d_*).
        zw = self.proj_w(w)
        zt = self.proj_t(t)
        zb = self.proj_b(b)
        if self.fusion == "cross_attention":
            tokens = torch.stack([zw, zt, zb], dim=1)        # (B, 3, d_v)
            attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
            tokens = tokens + attn_out
            tokens = tokens + self.ffn(tokens)
            fused = tokens.mean(dim=1)
        else:
            fused = self.fuse(torch.cat([zw, zt, zb], dim=-1))
        return self.out_norm(self.dropout(fused))
