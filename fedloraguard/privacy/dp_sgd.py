"""Per-client DP-SGD step (Algorithm 1, lines 6--8).

We implement the gradient-clip-and-Gaussian-noise primitive directly, as
opposed to wrapping Opacus, so that the federated runtime can compose
clipping over an arbitrary per-client objective without imposing the
Opacus ``PrivacyEngine.attach`` constraint on the model graph.

The resulting noised gradient ``\tilde g_i`` is what gets fed into the
SecAgg upload (`fedloraguard.federated.secure_agg.SecureAggregator`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch


@dataclass
class ClipNoiseConfig:
    clip_norm: float = 1.0
    noise_multiplier: float = 1.1
    enabled: bool = True


def _flatten(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def clip_and_noise(
    grads: List[torch.Tensor],
    cfg: ClipNoiseConfig,
    generator: torch.Generator | None = None,
) -> List[torch.Tensor]:
    """Clip the per-example gradient to ``S`` and add Gaussian noise sigma*S."""
    if not cfg.enabled or not grads:
        return [g.clone() for g in grads]
    flat = _flatten(grads)
    norm = torch.linalg.norm(flat) + 1e-12
    scale = torch.clamp(torch.tensor(cfg.clip_norm) / norm, max=1.0)
    clipped = [g * scale for g in grads]
    if cfg.noise_multiplier > 0.0:
        noised = []
        for g in clipped:
            noise = torch.randn(g.shape, device=g.device, generator=generator) * (
                cfg.noise_multiplier * cfg.clip_norm
            )
            noised.append(g + noise)
        return noised
    return clipped
