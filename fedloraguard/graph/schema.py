"""Heterogeneous continuous-time dynamic-graph schema.

Implements Definition 1 of the paper (Section 3.2):

    G(t) = (V(t), E(t), R, phi, psi)

with four node types  (adapter, base_model, contributor, application)
and six edge types    (contributes, derives_from, fine_tunes, deploys,
                       cites, co_uses).

Each adapter node carries the multimodal feature triple
(weight-modality, text-modality, behavioral-modality) on which the multimodal
encoder operates.  Edges are time-stamped to enable Bochner-style temporal
encoding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- type aliases --------------------------------------------------------------
NodeType = str
EdgeType = str
NodeId = Tuple[NodeType, int]            # (type, local index)
TimedEdge = Tuple[NodeId, NodeId, EdgeType, float]   # (src, dst, rel, t)

NODE_TYPES: Tuple[NodeType, ...] = (
    "adapter",
    "base_model",
    "contributor",
    "application",
)
EDGE_TYPES: Tuple[EdgeType, ...] = (
    "contributes",
    "derives_from",
    "fine_tunes",
    "deploys",
    "cites",
    "co_uses",
)


# --- adapter description -------------------------------------------------------
@dataclass
class AdapterRecord:
    """In-memory description of one LoRA adapter (paper Section 3.1).

    The `weights_BA` field stores per-layer (B, A) low-rank matrices but is
    optional: spectral signatures and Frobenius norms can be precomputed and
    passed via `weight_features` so that no raw weights ever leave the client.
    """
    adapter_id: str
    base_model: str
    contributor: str
    application: str
    rank: int
    upload_ts: float
    label: int                            # 0 benign, 1 backdoored
    weights_BA: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    weight_features: Optional[np.ndarray] = None
    text_features: Optional[np.ndarray] = None
    behavioral_features: Optional[np.ndarray] = None
    metadata: Dict[str, str] = field(default_factory=dict)


# --- in-memory graph container -------------------------------------------------
@dataclass
class HeteroDynamicGraph:
    """Plain in-memory heterogeneous CT-DG container.

    ``node_features[t]`` is a list of feature matrices indexed by node type,
    aligned with :data:`NODE_TYPES`.  ``edges`` is a flat list of timed edges;
    the :meth:`temporal_neighbors` helper indexes by destination node.
    """
    node_features: Dict[NodeType, np.ndarray]                # type -> (N_type, d_v)
    node_types: Tuple[NodeType, ...] = NODE_TYPES
    edge_types: Tuple[EdgeType, ...] = EDGE_TYPES
    edges: List[TimedEdge] = field(default_factory=list)
    node_ids: Dict[NodeType, List[str]] = field(default_factory=dict)
    labels: Dict[int, int] = field(default_factory=dict)     # adapter index -> label

    # ---- convenience accessors ----
    def num_nodes(self, t: Optional[NodeType] = None) -> int:
        if t is None:
            return sum(self.num_nodes(nt) for nt in self.node_types)
        return self.node_features.get(t, np.zeros((0, 0))).shape[0]

    def adjacency_by_dst(self) -> Dict[NodeId, List[TimedEdge]]:
        adj: Dict[NodeId, List[TimedEdge]] = {}
        for e in self.edges:
            adj.setdefault(e[1], []).append(e)
        for nid in adj:
            adj[nid].sort(key=lambda e: e[3])
        return adj

    def temporal_neighbors(
        self,
        node: NodeId,
        upper_time: float,
        k: int,
        adjacency: Optional[Dict[NodeId, List[TimedEdge]]] = None,
    ) -> List[TimedEdge]:
        """Return up to *k* most recent in-edges of `node` with timestamp <= upper_time."""
        adjacency = adjacency or self.adjacency_by_dst()
        all_in = [e for e in adjacency.get(node, []) if e[3] <= upper_time]
        return all_in[-k:]

    def split_by_marketplace(
        self,
        marketplace_of: Sequence[int],
    ) -> Dict[int, "HeteroDynamicGraph"]:
        """Partition the graph by marketplace assignment of adapter nodes.

        ``marketplace_of[i]`` is the marketplace id of adapter *i*; non-adapter
        nodes (base models, contributors, applications) are replicated across
        clients to model the public-side knowledge each marketplace already
        possesses about base models and contributors.
        """
        clients: Dict[int, HeteroDynamicGraph] = {}
        adapter_feat = self.node_features["adapter"]
        n_adapters = adapter_feat.shape[0]
        marketplace_ids = sorted(set(marketplace_of))
        for mid in marketplace_ids:
            mask = np.fromiter(
                ((m == mid) for m in marketplace_of),
                dtype=bool,
                count=n_adapters,
            )
            local_features = {
                nt: (
                    self.node_features[nt][mask] if nt == "adapter"
                    else self.node_features[nt]
                )
                for nt in self.node_types
            }
            local_indices = np.where(mask)[0]
            remap = {("adapter", int(g)): ("adapter", int(l))
                     for l, g in enumerate(local_indices)}
            local_edges: List[TimedEdge] = []
            for src, dst, rel, ts in self.edges:
                src_l = remap.get(src, src)
                dst_l = remap.get(dst, dst)
                # keep only edges incident to a local adapter or wholly between
                # public-side replicated nodes (base/contrib/app).
                if (src[0] == "adapter" and src not in remap) or (
                    dst[0] == "adapter" and dst not in remap
                ):
                    continue
                local_edges.append((src_l, dst_l, rel, ts))
            local_labels = {
                int(l): self.labels.get(int(g), 0)
                for l, g in enumerate(local_indices)
            }
            clients[mid] = HeteroDynamicGraph(
                node_features=local_features,
                edges=local_edges,
                node_ids={
                    nt: (
                        [self.node_ids["adapter"][i] for i in local_indices]
                        if nt == "adapter" and "adapter" in self.node_ids
                        else self.node_ids.get(nt, [])
                    )
                    for nt in self.node_types
                },
                labels=local_labels,
            )
        return clients
