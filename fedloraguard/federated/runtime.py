"""Single-process federated runtime that simulates ``N`` marketplace clients.

This is the in-process equivalent of the Flower runtime; we ship the Flower
adapter as ``runtime_flower.py`` for users running on real cross-silo
infrastructure.  The single-process loop is the canonical reproducibility
path so that the headline numbers can be regenerated on a single machine.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn

from ..graph.schema import HeteroDynamicGraph
from ..models.verifier import build_verifier
from ..privacy.certified_radius import certified_poisoning_radius
from ..privacy.dp_sgd import ClipNoiseConfig
from .client import FederatedClient, ClientConfig
from .sampling import build_query_batch
from .server import FederatedServer, ServerConfig


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _select_clients(t: int, num_clients: int, m: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed * 100003 + t)
    return list(rng.choice(num_clients, size=min(m, num_clients), replace=False))


def run_federated_training(
    cfg: Dict[str, Any],
    client_graphs: Dict[int, HeteroDynamicGraph],
    root_set_records,
    output_dir: str,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg["experiment"].get("device", "auto"))

    verifier = build_verifier(cfg).to(device)
    server_cfg = ServerConfig(
        num_clients=cfg["federated"]["num_clients"],
        sample_rate=cfg["federated"]["sampling_rate"],
        fltrust_enabled=cfg["federated"]["fltrust"]["enabled"],
        fltrust_threshold=cfg["federated"]["fltrust"]["score_threshold"],
        secure_agg=cfg["federated"]["secure_aggregation"]["enabled"],
        delta=cfg["privacy"]["target_delta"],
        noise_multiplier=cfg["privacy"]["noise_multiplier"],
    )
    server = FederatedServer(verifier, server_cfg, device=device)

    # Build clients ----------------------------------------------------------
    n_clients = cfg["federated"]["num_clients"]
    byzantine_ratio = float(cfg["federated"].get("byzantine_fraction", 0.0))
    n_byz = int(round(byzantine_ratio * n_clients))
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    byz_ids = set(rng.choice(n_clients, size=n_byz, replace=False).tolist()) if n_byz else set()

    clients: Dict[int, FederatedClient] = {}
    for cid in range(n_clients):
        cclient_cfg = ClientConfig(
            client_id=cid,
            is_byzantine=cid in byz_ids,
            local_epochs=cfg["federated"]["local_epochs"],
            batch_size=cfg["federated"]["batch_size"],
        )
        graph = client_graphs.get(cid, client_graphs[next(iter(client_graphs))])
        client_verifier = build_verifier(cfg)
        client_verifier.load_state_dict(server.get_state())
        clients[cid] = FederatedClient(cclient_cfg, client_verifier, graph, device=device)

    # Root set on server side -----------------------------------------------
    root_batch = build_query_batch(
        root_set_records, batch_size=cfg["federated"]["fltrust"]["root_set_size"], device=device
    ) if isinstance(root_set_records, HeteroDynamicGraph) else []

    criterion = nn.CrossEntropyLoss()
    clip_noise = ClipNoiseConfig(
        clip_norm=cfg["privacy"]["clip_norm"],
        noise_multiplier=cfg["privacy"]["noise_multiplier"],
        enabled=cfg["privacy"]["enabled"],
    )

    history: List[Dict[str, Any]] = []
    rounds = cfg["federated"]["rounds"]
    m = cfg["federated"]["clients_per_round"]
    lr = cfg["optim"]["lr"]

    for t in range(rounds):
        sampled = _select_clients(t, n_clients, m, cfg["experiment"]["seed"])
        # Push global state to sampled clients.
        global_state = server.get_state()
        for cid in sampled:
            clients[cid].set_state(global_state)

        # Reference gradient on server side via a fresh client copy.
        if root_batch:
            ref_client = next(iter(clients.values()))
            ref_grads = ref_client.reference_gradient(root_batch, criterion)
        else:
            ref_grads = None

        # Local updates ------------------------------------------------------
        client_grads: Dict[int, List[torch.Tensor]] = {}
        for cid in sampled:
            client_grads[cid] = clients[cid].local_update(criterion, lr, clip_noise)

        averaged = server.aggregate(client_grads, reference_grads=ref_grads)
        server.apply(averaged, lr)

        eps_t = server.epsilon()
        history.append({"round": t, "epsilon": eps_t,
                        "n_suspicious": len(server.suspicious_clients)})

        if eps_t >= cfg["privacy"]["target_epsilon"]:
            break

    # Save artifacts ---------------------------------------------------------
    ckpt = output_dir / "global.pt"
    torch.save(server.get_state(), ckpt)
    with (output_dir / "history.json").open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    return {"checkpoint": str(ckpt), "epsilon_T": server.epsilon(),
            "history": history}
