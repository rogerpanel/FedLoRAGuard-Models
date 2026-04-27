"""Krum-DyGFormer baseline: FedAvg + Krum aggregator over our DyGFormer-HGT
backbone (Section 6.2)."""
from __future__ import annotations

from typing import Dict, List

import torch


def krum_aggregate(client_grads: Dict[int, List[torch.Tensor]],
                   num_byzantine: int) -> List[torch.Tensor]:
    """Standard Krum (Blanchard et al., 2017): pick the client whose flattened
    gradient is closest to the (n - f - 2) nearest other clients."""
    ids = sorted(client_grads.keys())
    n = len(ids)
    f = max(0, num_byzantine)
    if n - f - 2 <= 0:
        # Fallback to mean if Krum is undefined.
        avg = [torch.stack([client_grads[c][i] for c in ids], dim=0).mean(dim=0)
               for i in range(len(client_grads[ids[0]]))]
        return avg
    flats = {c: torch.cat([t.reshape(-1) for t in client_grads[c]]) for c in ids}
    scores = {}
    for i in ids:
        dists = sorted(
            (float(torch.linalg.norm(flats[i] - flats[j]) ** 2) for j in ids if j != i)
        )
        scores[i] = sum(dists[: max(0, n - f - 2)])
    chosen = min(scores, key=scores.get)
    return [g.clone() for g in client_grads[chosen]]


class KrumDyGFormer:
    """Marker class -- the runtime is invoked by passing
    ``aggregator='krum'`` to ``scripts/train_federated.py``."""
    pass
