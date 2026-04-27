"""Temperature scaling (Guo et al., 2017) used by Section 6 of the paper.

A single scalar ``T`` is fit by NLL on a held-out validation set; this is the
calibration component referenced as our prior UCHGP methodology in the
manuscript.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TemperatureScaler(nn.Module):
    def __init__(self, init: float = 1.0) -> None:
        super().__init__()
        self.log_T = nn.Parameter(torch.log(torch.tensor(float(init))))

    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_T).clamp(min=1e-3)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature()


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 200,
    lr: float = 0.05,
) -> float:
    scaler = TemperatureScaler()
    opt = torch.optim.LBFGS([scaler.log_T], lr=lr, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = nll(scaler(logits), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(scaler.temperature().item())
