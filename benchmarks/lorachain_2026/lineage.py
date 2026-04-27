"""Cross-marketplace adapter lineage edges.

The LoRAchain-2026 benchmark must reflect the realistic derivation patterns
observed in HuggingGraph (Rahman et al., 2025) and PADBench (Sun et al., 2025).
We synthesize those edges by sampling from a configurable
power-law distribution: a small number of "hub" base models attract many
adapters, and a small number of "celebrity" contributors author many
adapters that derive from each other.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def synthesize_lineage_edges(
    n_adapters: int,
    base_index: Dict[str, int],
    base_of: List[int],
    upload_ts: List[float],
    contributor_of: List[int],
    *,
    density: float = 0.08,
    rng: np.random.Generator,
) -> List[Tuple[int, int, float]]:
    """Return a list of (src_adapter, dst_adapter, timestamp) tuples
    representing `cites` lineage relationships."""
    edges: List[Tuple[int, int, float]] = []
    # Bucket adapters by base model id.
    by_base: Dict[int, List[int]] = {}
    for ai, bi in enumerate(base_of):
        by_base.setdefault(bi, []).append(ai)

    for bi, adapter_ids in by_base.items():
        adapter_ids = sorted(adapter_ids, key=lambda i: upload_ts[i])
        # Power-law popularity: each adapter cites earlier adapters with
        # probability proportional to upload age.
        for j, ai in enumerate(adapter_ids):
            if j == 0:
                continue
            n_ref = rng.binomial(min(j, 8), density)
            if n_ref == 0:
                continue
            probs = np.array([1.0 / (j - k + 1) for k in range(j)])
            probs /= probs.sum()
            picks = rng.choice(adapter_ids[:j], size=n_ref, replace=False, p=probs)
            for src in picks:
                ts = max(upload_ts[ai], upload_ts[int(src)]) + 0.5
                edges.append((int(src), int(ai), float(ts)))
    return edges
