"""Behavioral-modality features for an adapter (Section 4.1):
download counts, monthly active deployments, citation-graph centralities.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


class BehavioralEncoder:
    def __init__(self, log_normalize: bool = True, dim: int = 16) -> None:
        self.log_normalize = log_normalize
        self.dim = dim

    def encode(self, stats: Dict[str, float]) -> np.ndarray:
        vals = np.zeros(self.dim, dtype=np.float32)
        keys = sorted(stats.keys())
        for i, k in enumerate(keys[: self.dim]):
            x = float(stats[k])
            vals[i] = float(np.log1p(max(x, 0.0))) if self.log_normalize else x
        if vals.size:
            scale = float(np.linalg.norm(vals)) + 1e-9
            vals /= scale
        return vals
