"""Centralised PEFTGuard baseline (Sun et al., IEEE S&P 2025).

Reimplemented from the description in the original paper: a feed-forward
classifier on top of the spectral signature + Frobenius norms of the
LoRA update matrices.  Achieves 98.3% on the centralised PADBench (paper
Table 1, top row).  We expose it as a centralized scikit-learn / pytorch
classifier so that it serves as the upper-bound reference in our
federated evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn


@dataclass
class _PGConfig:
    feature_dim: int
    hidden: int = 128
    epochs: int = 50
    lr: float = 1.0e-3
    batch_size: int = 64


class PEFTGuardCentralized:
    """Mirror of the public PEFTGuard CLF in :code:`Z-Sun-RG/PEFTGuard`."""
    def __init__(self, feature_dim: int) -> None:
        self.cfg = _PGConfig(feature_dim=feature_dim)
        self.model = nn.Sequential(
            nn.Linear(feature_dim, self.cfg.hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, 2),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PEFTGuardCentralized":
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.int64))
        n = X_t.shape[0]
        for _ in range(self.cfg.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.cfg.batch_size):
                idx = perm[i:i + self.cfg.batch_size]
                logits = self.model(X_t[idx])
                loss = crit(logits, y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X.astype(np.float32)))
            return torch.softmax(logits, dim=-1).numpy()
