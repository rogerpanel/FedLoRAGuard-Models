"""Bochner-style functional time encoding (TGAT, Xu et al., 2020) and
temporal-neighbor sampling utilities used by DyGFormer (Yu et al., 2023).
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


class BochnerTimeEncoding(nn.Module):
    r"""Functional time encoding ``Phi(t) = cos(omega * t + phi)``.

    Parameters
    ----------
    dim : int
        Output dimension; must be even-ish so that we can mix sin/cos.
    learnable : bool
        Whether the frequencies and phases are trainable (DyGFormer default).
    """

    def __init__(self, dim: int = 32, learnable: bool = True) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("BochnerTimeEncoding requires dim > 0")
        self.dim = dim
        # log-spaced initial frequencies (cf. TGAT default).
        freqs = torch.from_numpy(
            1.0 / np.power(10_000, np.linspace(0, 9, dim))
        ).float()
        phases = torch.zeros(dim)
        if learnable:
            self.omega = nn.Parameter(freqs)
            self.phi = nn.Parameter(phases)
        else:
            self.register_buffer("omega", freqs)
            self.register_buffer("phi", phases)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        # dt: (..., 1) -> output (..., dim)
        return torch.cos(dt.unsqueeze(-1) * self.omega + self.phi)


def sample_temporal_neighbors(
    edges_in: Sequence[Tuple],
    upper_time: float,
    k: int,
) -> List[Tuple]:
    """Return the *k* most recent edges whose timestamp <= upper_time."""
    filtered = [e for e in edges_in if e[-1] <= upper_time]
    return filtered[-k:]
