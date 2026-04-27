"""Federated-Averaging MLP baseline (Section 6.2, ``FedAvg-MLP``)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn


@dataclass
class _Cfg:
    feature_dim: int
    hidden: int = 128
    rounds: int = 100
    lr: float = 1.0e-3


class FedAvgMLP:
    def __init__(self, feature_dim: int) -> None:
        self.cfg = _Cfg(feature_dim=feature_dim)
        self.model = nn.Sequential(
            nn.Linear(feature_dim, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, 2),
        )

    def fit_federated(self, client_data: Dict[int, tuple]) -> "FedAvgMLP":
        crit = nn.CrossEntropyLoss()
        for _ in range(self.cfg.rounds):
            states = []
            for X, y in client_data.values():
                local = nn.Sequential(*[type(m)(*self._init_args(m)) for m in self.model])
                local.load_state_dict(self.model.state_dict())
                opt = torch.optim.SGD(local.parameters(), lr=self.cfg.lr)
                logits = local(torch.from_numpy(X.astype(np.float32)))
                loss = crit(logits, torch.from_numpy(y.astype(np.int64)))
                opt.zero_grad(); loss.backward(); opt.step()
                states.append(local.state_dict())
            agg = {k: torch.stack([s[k] for s in states], dim=0).mean(dim=0) for k in states[0]}
            self.model.load_state_dict(agg)
        return self

    def _init_args(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            return (m.in_features, m.out_features)
        return ()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X.astype(np.float32)))
            return torch.softmax(logits, dim=-1).numpy()
