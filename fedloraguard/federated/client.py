"""Per-marketplace client (`C_i` in Algorithm 1).

A client owns:
  * a local ``HeteroDynamicGraph`` slice with the adapter weights it hosts,
  * a copy of the global verifier parameters,
  * a private random number generator so that DP-SGD noise is reproducible
    across runs but uncorrelated across clients.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from ..graph.schema import HeteroDynamicGraph
from ..models.verifier import FedLoRAGuardVerifier
from ..privacy.dp_sgd import ClipNoiseConfig, clip_and_noise
from .sampling import build_query_batch


@dataclass
class ClientConfig:
    client_id: int
    is_byzantine: bool = False
    local_epochs: int = 1
    batch_size: int = 64


class FederatedClient:
    def __init__(
        self,
        cfg: ClientConfig,
        verifier: FedLoRAGuardVerifier,
        graph: HeteroDynamicGraph,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.verifier = verifier.to(device)
        self.graph = graph
        self.device = torch.device(device)
        self.generator = torch.Generator(device=str(device)).manual_seed(int(cfg.client_id) * 9973)

    # -------- model parameter sync ------------------------------------------
    def set_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.verifier.load_state_dict(state_dict, strict=True)

    def get_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.verifier.state_dict().items()}

    # -------- local update --------------------------------------------------
    def local_update(
        self,
        criterion: nn.Module,
        lr: float,
        clip_noise: ClipNoiseConfig,
    ) -> List[torch.Tensor]:
        """Return the noised gradient ``\tilde g_i`` (Algorithm 1 lines 5--7)."""
        self.verifier.train()
        params = list(self.verifier.parameters())
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        for _ in range(self.cfg.local_epochs):
            batch = build_query_batch(self.graph, self.cfg.batch_size, device=self.device)
            if not batch:
                continue
            logits = self.verifier.forward_batch(batch)
            labels = torch.tensor(
                [b["label"] for b in batch], dtype=torch.long, device=self.device
            )
            if self.cfg.is_byzantine:
                # Label-flip Byzantine model (Definition 2 of the paper).
                labels = 1 - labels
            loss = criterion(logits, labels)
            loss.backward()

        grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                 for p in params]
        noised = clip_and_noise(grads, clip_noise, generator=self.generator)
        return noised

    def reference_gradient(self, root_set: List[Dict[str, Any]], criterion: nn.Module) -> List[torch.Tensor]:
        """Compute the FLTrust root gradient on a server-supplied vetted batch."""
        self.verifier.train()
        for p in self.verifier.parameters():
            if p.grad is not None:
                p.grad.zero_()
        if not root_set:
            return [torch.zeros_like(p) for p in self.verifier.parameters()]
        logits = self.verifier.forward_batch(root_set)
        labels = torch.tensor([b["label"] for b in root_set], dtype=torch.long, device=self.device)
        loss = criterion(logits, labels)
        loss.backward()
        return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in self.verifier.parameters()]
