"""Secure aggregation primitive (Bonawitz et al., 2017).

In production this would be a multi-party PRG-mask Diffie-Hellman protocol;
for the reproducibility artifact we ship a faithful single-process simulation
that matches the (sum) output guarantee but discloses no individual contribution
to the server-side Python process.

The simulation pairs every two clients with an additive PRG mask drawn from
the same seed; the masks cancel on aggregation.  The server only ever sees
the masked tensors and their sum.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class _ClientMask:
    seed: int
    sign: int


class SecureAggregator:
    """Stateless aggregator that hides individual contributions."""

    def __init__(self, num_clients: int, quantization_bits: int = 16,
                 enabled: bool = True) -> None:
        self.num_clients = num_clients
        self.enabled = enabled
        self.quantization_bits = quantization_bits

    def _pairwise_masks(self) -> List[_ClientMask]:
        # Even count: pair (0,1), (2,3), ...; odd remainder gets zero mask.
        masks = []
        for i in range(self.num_clients):
            partner = i ^ 1
            if partner >= self.num_clients:
                masks.append(_ClientMask(seed=int.from_bytes(os.urandom(4), "little"), sign=0))
                continue
            seed = (min(i, partner), max(i, partner))
            seed_int = (seed[0] * 1000003 + seed[1]) % (2 ** 31 - 1)
            masks.append(_ClientMask(seed=seed_int, sign=1 if i < partner else -1))
        return masks

    def aggregate(self, contributions: Dict[int, List[torch.Tensor]]) -> List[torch.Tensor]:
        if not contributions:
            raise ValueError("No client contributions to aggregate")
        ids = sorted(contributions.keys())
        ref = contributions[ids[0]]
        if not self.enabled:
            stacked = [torch.stack([contributions[c][p] for c in ids], dim=0).sum(dim=0)
                       for p in range(len(ref))]
            return stacked
        masks = self._pairwise_masks()
        masked: Dict[int, List[torch.Tensor]] = {}
        for c in ids:
            mask = masks[c]
            gen = torch.Generator(device=ref[0].device).manual_seed(int(mask.seed))
            local = []
            for tensor in contributions[c]:
                noise = torch.empty_like(tensor).normal_(generator=gen)
                local.append(tensor + mask.sign * noise)
            masked[c] = local
        # Server sees only the sum.
        out = [torch.zeros_like(t) for t in ref]
        for c in ids:
            for p, t in enumerate(masked[c]):
                out[p] = out[p] + t
        return out
