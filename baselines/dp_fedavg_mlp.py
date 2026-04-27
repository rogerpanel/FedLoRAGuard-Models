"""DP-FedAvg-MLP baseline (Section 6.2): client-level DP-SGD on top of FedAvg-MLP."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn

from fedloraguard.privacy.dp_sgd import ClipNoiseConfig, clip_and_noise

from .fedavg_mlp import FedAvgMLP, _Cfg


class DPFedAvgMLP(FedAvgMLP):
    def __init__(self, feature_dim: int, clip_norm: float = 1.0,
                 noise_multiplier: float = 1.1) -> None:
        super().__init__(feature_dim)
        self.dp_cfg = ClipNoiseConfig(clip_norm=clip_norm, noise_multiplier=noise_multiplier)

    def fit_federated(self, client_data: Dict[int, tuple]) -> "DPFedAvgMLP":
        crit = nn.CrossEntropyLoss()
        for _ in range(self.cfg.rounds):
            grads_list = []
            for X, y in client_data.values():
                local = nn.Sequential(
                    nn.Linear(self.cfg.feature_dim, self.cfg.hidden),
                    nn.GELU(),
                    nn.Linear(self.cfg.hidden, 2),
                )
                local.load_state_dict(self.model.state_dict())
                logits = local(torch.from_numpy(X.astype(np.float32)))
                loss = crit(logits, torch.from_numpy(y.astype(np.int64)))
                loss.backward()
                grads = [p.grad.detach().clone() if p.grad is not None
                         else torch.zeros_like(p) for p in local.parameters()]
                grads_list.append(clip_and_noise(grads, self.dp_cfg))
            avg = [torch.stack([g[i] for g in grads_list], dim=0).mean(dim=0)
                   for i in range(len(grads_list[0]))]
            with torch.no_grad():
                for p, g in zip(self.model.parameters(), avg):
                    p.add_(g, alpha=-self.cfg.lr)
        return self
