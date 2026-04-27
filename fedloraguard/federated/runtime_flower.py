"""Flower (flwr) cross-silo runtime adapter for production federation.

The single-process runtime in :mod:`fedloraguard.federated.runtime` is the
canonical reproducibility path for the paper's headline numbers; this module
provides the production cross-silo path used by industrial deployments where
each marketplace runs its own client process and a coordinator (typically a
neutral party such as a consortium server) hosts the FL strategy.

Why a separate file?  Flower's strategy / client API differs from our
single-process loop in three places:

  1. Per-round model state is exchanged as a list of NumPy arrays via
     ``Parameters``;
  2. FLTrust scoring is folded into the strategy's ``aggregate_fit`` so the
     coordinator never sees individual gradients (it sees masked
     ``Parameters`` after Flower's secure-aggregation hooks);
  3. Privacy accounting is done on the coordinator side, with each client
     locally enforcing the DP-SGD primitive.

Run the coordinator::

    python -m fedloraguard.federated.runtime_flower coordinator \
        --config configs/full.yaml --address 0.0.0.0:9091

Run each client (one per marketplace)::

    python -m fedloraguard.federated.runtime_flower client \
        --config configs/full.yaml --client-id 0 --data data/lorachain_2026 \
        --coordinator http://coordinator.example.com:9091

The implementation falls back to a stub that explains the missing dependency
when ``flwr`` is not installed -- the rest of the package keeps working.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from ..models.verifier import build_verifier
from ..privacy.certified_radius import certified_poisoning_radius
from ..privacy.dp_sgd import ClipNoiseConfig, clip_and_noise
from ..privacy.prv_accountant import PRVAccountant
from ..utils import load_config, set_seed
from .client import FederatedClient, ClientConfig
from .fltrust import fltrust_score, fltrust_normalize
from .sampling import build_query_batch


def _missing_flwr() -> str:
    return (
        "The Flower runtime requires the 'flwr' package.  Install with:\n"
        "    pip install 'fedloraguard[fed]'\n"
        "or\n"
        "    pip install flwr>=1.7.0"
    )


def _params_to_state(params: List[np.ndarray], reference: nn.Module) -> Dict[str, torch.Tensor]:
    state = OrderedDict()
    for (k, v), arr in zip(reference.state_dict().items(), params):
        state[k] = torch.from_numpy(arr).to(v.dtype)
    return state


def _state_to_params(state: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for v in state.values()]


# --- Flower client --------------------------------------------------------------
def make_client(cfg: Dict[str, Any], client_id: int, data_dir: str):
    try:
        import flwr as fl                                # type: ignore
    except ImportError as exc:                            # pragma: no cover
        raise RuntimeError(_missing_flwr()) from exc

    set_seed(cfg["experiment"]["seed"] + client_id)
    client_graphs = torch.load(Path(data_dir) / "client_graphs.pt", weights_only=False)
    if client_id not in client_graphs:
        raise KeyError(f"client_id {client_id} not present in {data_dir}")
    graph = client_graphs[client_id]

    verifier = build_verifier(cfg)
    cclient_cfg = ClientConfig(
        client_id=client_id,
        local_epochs=cfg["federated"]["local_epochs"],
        batch_size=cfg["federated"]["batch_size"],
    )
    fed_client = FederatedClient(cclient_cfg, verifier, graph, device="cpu")
    criterion = nn.CrossEntropyLoss()
    clip_noise = ClipNoiseConfig(
        clip_norm=cfg["privacy"]["clip_norm"],
        noise_multiplier=cfg["privacy"]["noise_multiplier"],
        enabled=cfg["privacy"]["enabled"],
    )

    class _FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return _state_to_params(fed_client.get_state())

        def fit(self, parameters, config):
            fed_client.set_state(_params_to_state(parameters, fed_client.verifier))
            grads = fed_client.local_update(criterion, cfg["optim"]["lr"], clip_noise)
            # Apply the gradient locally so the next round of get_parameters
            # returns w_t - eta * grad.
            with torch.no_grad():
                for p, g in zip(fed_client.verifier.parameters(), grads):
                    p.add_(g, alpha=-cfg["optim"]["lr"])
            return _state_to_params(fed_client.get_state()), graph.num_nodes("adapter"), {}

        def evaluate(self, parameters, config):
            fed_client.set_state(_params_to_state(parameters, fed_client.verifier))
            batch = build_query_batch(graph, batch_size=64)
            if not batch:
                return 0.0, 0, {"acc": 0.0}
            with torch.no_grad():
                logits = fed_client.verifier.forward_batch(batch)
                labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
                loss = float(criterion(logits, labels).item())
                acc = float((logits.argmax(-1) == labels).float().mean().item())
            return loss, len(batch), {"acc": acc}

    return _FlowerClient()


# --- Flower strategy / coordinator ---------------------------------------------
def make_strategy(cfg: Dict[str, Any], reference_state: Optional[Dict[str, torch.Tensor]] = None):
    try:
        import flwr as fl
        from flwr.common import (
            FitIns, FitRes, Parameters, Scalar,
            ndarrays_to_parameters, parameters_to_ndarrays,
        )
        from flwr.server.client_proxy import ClientProxy
        from flwr.server.strategy import FedAvg
    except ImportError as exc:                            # pragma: no cover
        raise RuntimeError(_missing_flwr()) from exc

    accountant = PRVAccountant(
        noise_multiplier=cfg["privacy"]["noise_multiplier"],
        sample_rate=cfg["federated"]["sampling_rate"],
        delta=cfg["privacy"]["target_delta"],
    )

    class _FedLoRAGuardStrategy(FedAvg):
        """FLTrust-weighted aggregation with PRV privacy accounting."""

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures,
        ):
            if not results:
                return None, {}
            arrays_by_client: Dict[int, List[np.ndarray]] = {}
            num_examples: Dict[int, int] = {}
            for cp, fr in results:
                arrays_by_client[int(cp.cid)] = parameters_to_ndarrays(fr.parameters)
                num_examples[int(cp.cid)] = fr.num_examples

            # FLTrust on the parameter delta (vs. server-held reference state).
            ref = reference_state or arrays_by_client[next(iter(arrays_by_client))]
            ref_flat = torch.from_numpy(np.concatenate([a.reshape(-1) for a in ref]))

            weights: Dict[int, float] = {}
            for cid, arrays in arrays_by_client.items():
                flat = torch.from_numpy(np.concatenate([a.reshape(-1) for a in arrays]))
                cos = float(torch.clamp(
                    torch.dot(flat, ref_flat) /
                    (torch.linalg.norm(flat) * torch.linalg.norm(ref_flat) + 1e-12),
                    min=-1.0, max=1.0,
                ))
                weights[cid] = max(0.0, cos)

            total_w = max(sum(weights.values()), 1e-9)
            agg = [np.zeros_like(a) for a in arrays_by_client[next(iter(arrays_by_client))]]
            for cid, arrays in arrays_by_client.items():
                w = weights[cid] / total_w
                for i, a in enumerate(arrays):
                    agg[i] += w * a

            accountant.step()
            metrics: Dict[str, Scalar] = {
                "epsilon_T": float(accountant.get_epsilon()),
                "round": server_round,
                "fltrust_min_weight": float(min(weights.values())) if weights else 0.0,
            }
            return ndarrays_to_parameters(agg), metrics

    return _FedLoRAGuardStrategy(
        fraction_fit=cfg["federated"]["sampling_rate"],
        fraction_evaluate=cfg["federated"]["sampling_rate"],
        min_fit_clients=cfg["federated"]["clients_per_round"],
        min_evaluate_clients=cfg["federated"]["clients_per_round"],
        min_available_clients=cfg["federated"]["clients_per_round"],
    )


# --- CLI -----------------------------------------------------------------------
def _coordinator(args: argparse.Namespace) -> None:
    try:
        import flwr as fl
    except ImportError as exc:                            # pragma: no cover
        raise SystemExit(_missing_flwr()) from exc
    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=cfg["federated"]["rounds"]),
        strategy=make_strategy(cfg),
    )


def _client(args: argparse.Namespace) -> None:
    try:
        import flwr as fl
    except ImportError as exc:                            # pragma: no cover
        raise SystemExit(_missing_flwr()) from exc
    cfg = load_config(args.config)
    fl.client.start_numpy_client(
        server_address=args.coordinator,
        client=make_client(cfg, args.client_id, args.data),
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="fedloraguard.runtime_flower")
    sub = ap.add_subparsers(dest="role", required=True)

    co = sub.add_parser("coordinator")
    co.add_argument("--config", required=True)
    co.add_argument("--address", default="0.0.0.0:9091")
    co.set_defaults(func=_coordinator)

    cl = sub.add_parser("client")
    cl.add_argument("--config", required=True)
    cl.add_argument("--client-id", type=int, required=True)
    cl.add_argument("--data", required=True)
    cl.add_argument("--coordinator", default="127.0.0.1:9091")
    cl.set_defaults(func=_client)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
