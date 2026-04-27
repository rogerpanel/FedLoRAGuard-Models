"""Mini-batch sampler that turns a :class:`HeteroDynamicGraph` slice into the
list-of-dicts representation consumed by :meth:`FedLoRAGuardVerifier.forward_batch`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..graph.schema import HeteroDynamicGraph


def _node_feat(graph: HeteroDynamicGraph, nid) -> np.ndarray:
    return graph.node_features[nid[0]][nid[1]]


def build_query_batch(
    graph: HeteroDynamicGraph,
    batch_size: int,
    *,
    device: torch.device | str = "cpu",
    rng: Optional[np.random.Generator] = None,
    upper_time: Optional[float] = None,
    num_neighbors: int = 32,
) -> List[Dict[str, Any]]:
    rng = rng or np.random.default_rng()
    n_a = graph.num_nodes("adapter")
    if n_a == 0:
        return []
    chosen = rng.choice(n_a, size=min(batch_size, n_a), replace=False)
    adjacency = graph.adjacency_by_dst()
    out: List[Dict[str, Any]] = []
    for ai in chosen.tolist():
        node = ("adapter", ai)
        if upper_time is None:
            t_query = max((e[3] for e in adjacency.get(node, [])), default=0.0)
        else:
            t_query = upper_time
        neighbors = graph.temporal_neighbors(node, t_query, num_neighbors, adjacency=adjacency)
        nbr_feats = []
        nbr_types: List[str] = []
        nbr_rels: List[str] = []
        nbr_ts: List[float] = []
        for src, _dst, rel, ts in neighbors:
            nbr_feats.append(_node_feat(graph, src))
            nbr_types.append(src[0])
            nbr_rels.append(rel)
            nbr_ts.append(ts)
        if nbr_feats:
            nbr_feats_t = torch.from_numpy(np.stack(nbr_feats)).float().to(device)
        else:
            nbr_feats_t = torch.zeros(0, graph.node_features["adapter"].shape[1], device=device)
        rel_times_t = torch.tensor(
            [t_query - t for t in nbr_ts], dtype=torch.float32, device=device
        )
        out.append({
            "query_feat": torch.from_numpy(_node_feat(graph, node)).float().to(device),
            "neighbor_feats": nbr_feats_t,
            "neighbor_types": nbr_types,
            "relations": nbr_rels,
            "rel_times": rel_times_t,
            "label": int(graph.labels.get(ai, 0)),
            "adapter_id": graph.node_ids.get("adapter", [str(ai)])[ai],
            "query_type": "adapter",
        })
    return out
