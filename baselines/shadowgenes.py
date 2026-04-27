"""ShadowGenes baseline (Schulz & Evans, 2025).

Graph-based model-genealogy verification using a static computational-graph
fingerprint. We approximate the original "subgraph signature" matcher with a
random-walk fingerprint comparison (cosine similarity over hashed edge
sequences).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import hashlib
import numpy as np


@dataclass
class _SGConfig:
    walk_length: int = 8
    num_walks: int = 32
    hash_dim: int = 256


class ShadowGenesCentralized:
    def __init__(self) -> None:
        self.cfg = _SGConfig()
        self.benign_fingerprints: List[np.ndarray] = []

    def _walk_fingerprint(self, signature: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(int(np.abs(signature).sum() * 1000) % (2 ** 31))
        v = rng.choice(signature.shape[0], size=self.cfg.walk_length * self.cfg.num_walks)
        out = np.zeros(self.cfg.hash_dim, dtype=np.float32)
        for i in v:
            h = int(hashlib.sha1(int(signature[int(i)]).to_bytes(4, "little", signed=False)).hexdigest(), 16)
            out[h % self.cfg.hash_dim] += 1.0
        out /= np.linalg.norm(out) + 1e-9
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ShadowGenesCentralized":
        for i, label in enumerate(y):
            if label == 0:
                self.benign_fingerprints.append(self._walk_fingerprint(X[i]))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.benign_fingerprints:
            raise RuntimeError("ShadowGenes was not fitted with any benign references.")
        ref = np.stack(self.benign_fingerprints)
        out = np.zeros((X.shape[0], 2), dtype=np.float32)
        for i in range(X.shape[0]):
            fp = self._walk_fingerprint(X[i])
            sim = float((ref @ fp).max())
            p_benign = float(min(max(sim, 0.0), 1.0))
            out[i, 0] = p_benign
            out[i, 1] = 1.0 - p_benign
        return out
