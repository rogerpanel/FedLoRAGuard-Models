"""FLTrust trust-bootstrapping (Cao et al., 2021, NDSS).

Each client gradient `g_i` is scored by its cosine similarity to the
server-computed reference gradient `g_0` on a small vetted root set `R_0`,
clipped to the magnitude of `g_0`, and weighted by ReLU(cos(g_i, g_0)).
"""
from __future__ import annotations

from typing import Iterable, List

import torch


def _flatten(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def fltrust_score(client_grads: List[torch.Tensor], reference_grads: List[torch.Tensor]) -> float:
    g_i = _flatten(client_grads)
    g_0 = _flatten(reference_grads)
    n_i = torch.linalg.norm(g_i) + 1e-12
    n_0 = torch.linalg.norm(g_0) + 1e-12
    cos = float(torch.clamp(torch.dot(g_i, g_0) / (n_i * n_0), min=-1.0, max=1.0))
    return max(0.0, cos)


def fltrust_normalize(client_grads: List[torch.Tensor], reference_grads: List[torch.Tensor]) -> List[torch.Tensor]:
    """Clip the L2 norm of `client_grads` to that of `reference_grads`."""
    g_i = _flatten(client_grads)
    g_0 = _flatten(reference_grads)
    n_i = torch.linalg.norm(g_i) + 1e-12
    n_0 = torch.linalg.norm(g_0) + 1e-12
    scale = torch.clamp(n_0 / n_i, max=1.0)
    return [g * scale for g in client_grads]
