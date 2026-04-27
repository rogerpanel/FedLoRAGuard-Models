"""Build a :class:`HeteroDynamicGraph` from a list of :class:`AdapterRecord`.

Edges are synthesised according to four canonical relations recorded in the
paper (Section 3.2) plus two relational closures:

  * ``contributes(contributor, adapter)``   -- one edge per adapter
  * ``derives_from(adapter, base_model)``   -- one per adapter
  * ``fine_tunes(adapter, application)``    -- task-targeting
  * ``deploys(application, adapter)``       -- adoption
  * ``cites(adapter, adapter)``             -- lineage edges (sparse)
  * ``co_uses(application, application)``   -- shared base model

The exact ``cites`` density is configurable from the ``benchmark`` block of
the YAML config.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .schema import (
    EDGE_TYPES,
    NODE_TYPES,
    AdapterRecord,
    HeteroDynamicGraph,
    NodeId,
    TimedEdge,
)


def _index_unique(values: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        if v not in out:
            out[v] = len(out)
    return out


def build_graph_from_records(
    records: Sequence[AdapterRecord],
    feature_dims: Dict[str, int],
    *,
    lineage_density: float = 0.08,
    citation_density: float = 0.04,
    rng: Optional[np.random.Generator] = None,
) -> HeteroDynamicGraph:
    """Build the heterogeneous CT-DG from a list of adapter records.

    The function honors the ``feature_dims`` block of the config: missing
    modalities are zero-padded to the right dimension so that downstream
    encoders see a consistent shape.
    """
    rng = rng or np.random.default_rng(0)

    # Index nodes -------------------------------------------------------------
    base_ix = _index_unique([r.base_model for r in records])
    contrib_ix = _index_unique([r.contributor for r in records])
    app_ix = _index_unique([r.application for r in records])

    n_a = len(records)
    d_w = feature_dims.get("weight", 16)
    d_t = feature_dims.get("text", 16)
    d_b = feature_dims.get("behavioral", 4)
    raw_dim = d_w + d_t + d_b      # raw modality concatenation width

    def _pad(x: Optional[np.ndarray], dim: int) -> np.ndarray:
        if x is None:
            return np.zeros(dim, dtype=np.float32)
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] >= dim:
            return x[:dim]
        return np.concatenate([x, np.zeros(dim - x.shape[0], dtype=np.float32)])

    adapter_feats = np.stack(
        [
            np.concatenate([
                _pad(r.weight_features, d_w),
                _pad(r.text_features, d_t),
                _pad(r.behavioral_features, d_b),
            ])
            for r in records
        ]
    )
    assert adapter_feats.shape[1] == raw_dim, (
        f"adapter raw feature dim {adapter_feats.shape[1]} != weight+text+beh {raw_dim}"
    )

    # All node types must share the same raw width so that the multimodal
    # encoder in the verifier can slice them by modality (Eq. 3).  Non-adapter
    # nodes carry zero in the modality slots they do not produce.
    base_feats = rng.normal(0.0, 0.05, size=(len(base_ix), raw_dim)).astype(np.float32)
    contrib_feats = rng.normal(0.0, 0.05, size=(len(contrib_ix), raw_dim)).astype(np.float32)
    app_feats = rng.normal(0.0, 0.05, size=(len(app_ix), raw_dim)).astype(np.float32)

    edges: List[TimedEdge] = []

    # contributes / derives_from / fine_tunes / deploys
    for ai, r in enumerate(records):
        ts = float(r.upload_ts)
        edges.append(((("contributor", contrib_ix[r.contributor])), ("adapter", ai), "contributes", ts))
        edges.append((("adapter", ai), ("base_model", base_ix[r.base_model]), "derives_from", ts))
        edges.append((("adapter", ai), ("application", app_ix[r.application]), "fine_tunes", ts))
        edges.append((("application", app_ix[r.application]), ("adapter", ai), "deploys", ts + 1.0))

    # citation edges -- sparse lineage among adapters with the same base model
    by_base: Dict[int, List[int]] = {}
    for ai, r in enumerate(records):
        by_base.setdefault(base_ix[r.base_model], []).append(ai)
    for base_id, adapter_ids in by_base.items():
        adapter_ids = sorted(adapter_ids, key=lambda i: records[i].upload_ts)
        for j, ai in enumerate(adapter_ids):
            n_cites = rng.binomial(min(j, 8), lineage_density)
            if n_cites <= 0:
                continue
            srcs = rng.choice(adapter_ids[:j], size=n_cites, replace=False)
            for src in srcs:
                ts = max(records[ai].upload_ts, records[int(src)].upload_ts) + 0.5
                edges.append((("adapter", int(src)), ("adapter", ai), "cites", float(ts)))

    # co_uses among applications sharing a base model
    apps_by_base: Dict[int, List[int]] = {}
    for r in records:
        apps_by_base.setdefault(base_ix[r.base_model], []).append(app_ix[r.application])
    for base_id, app_ids in apps_by_base.items():
        unique_app_ids = list(set(app_ids))
        if len(unique_app_ids) < 2:
            continue
        n_pairs = max(1, int(citation_density * len(unique_app_ids)))
        for _ in range(n_pairs):
            a, b = rng.choice(unique_app_ids, size=2, replace=False)
            edges.append((("application", int(a)), ("application", int(b)), "co_uses", 0.0))

    node_features = {
        "adapter": adapter_feats,
        "base_model": base_feats,
        "contributor": contrib_feats,
        "application": app_feats,
    }
    node_ids = {
        "adapter": [r.adapter_id for r in records],
        "base_model": list(base_ix.keys()),
        "contributor": list(contrib_ix.keys()),
        "application": list(app_ix.keys()),
    }
    labels = {ai: int(r.label) for ai, r in enumerate(records)}
    return HeteroDynamicGraph(
        node_features=node_features,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        edges=edges,
        node_ids=node_ids,
        labels=labels,
    )
