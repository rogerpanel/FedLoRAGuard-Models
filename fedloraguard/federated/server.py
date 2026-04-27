"""Server-side aggregation logic (Algorithm 1, lines 10--14).

Holds the global verifier parameters and the FLTrust root set; orchestrates
secure aggregation, trust scoring, weighted averaging, and privacy accounting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from ..models.verifier import FedLoRAGuardVerifier
from ..privacy.prv_accountant import PRVAccountant
from .fltrust import fltrust_score, fltrust_normalize
from .secure_agg import SecureAggregator


@dataclass
class ServerConfig:
    num_clients: int
    sample_rate: float = 0.2
    fltrust_enabled: bool = True
    fltrust_threshold: float = 0.0
    secure_agg: bool = True
    delta: float = 1e-5
    noise_multiplier: float = 1.1


class FederatedServer:
    def __init__(
        self,
        verifier: FedLoRAGuardVerifier,
        cfg: ServerConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.verifier = verifier.to(device)
        self.device = device
        self.cfg = cfg
        self.aggregator = SecureAggregator(num_clients=cfg.num_clients, enabled=cfg.secure_agg)
        self.accountant = PRVAccountant(noise_multiplier=cfg.noise_multiplier,
                                        sample_rate=cfg.sample_rate, delta=cfg.delta)
        self.suspicious_clients: List[int] = []

    def get_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.verifier.state_dict().items()}

    def aggregate(
        self,
        client_grads: Dict[int, List[torch.Tensor]],
        reference_grads: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        if not client_grads:
            return [torch.zeros_like(p) for p in self.verifier.parameters()]

        # FLTrust scoring (paper line 11) + clip-to-reference (Cao et al., 2021).
        weights: Dict[int, float] = {}
        normalized: Dict[int, List[torch.Tensor]] = {}
        if self.cfg.fltrust_enabled and reference_grads is not None:
            for cid, g_i in client_grads.items():
                w = fltrust_score(g_i, reference_grads)
                if w <= self.cfg.fltrust_threshold:
                    self.suspicious_clients.append(cid)
                    weights[cid] = 0.0
                    normalized[cid] = [torch.zeros_like(g) for g in g_i]
                    continue
                weights[cid] = w
                normalized[cid] = fltrust_normalize(g_i, reference_grads)
        else:
            for cid, g_i in client_grads.items():
                weights[cid] = 1.0
                normalized[cid] = g_i

        # Secure aggregation: hides individual contributions; outputs sum only.
        contributions = {
            cid: [w * g for g in normalized[cid]] for cid, w in weights.items()
        }
        summed = self.aggregator.aggregate(contributions)
        total_weight = max(sum(weights.values()), 1e-9)
        averaged = [g / total_weight for g in summed]
        # Privacy account a single composition step per round.
        self.accountant.step()
        return averaged

    def apply(self, averaged_grad: List[torch.Tensor], lr: float) -> None:
        with torch.no_grad():
            for p, g in zip(self.verifier.parameters(), averaged_grad):
                p.add_(g, alpha=-lr)

    def epsilon(self) -> float:
        return self.accountant.get_epsilon()
