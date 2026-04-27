"""Text-modality features (model card, README, contributor profile).

Production path is a frozen sentence-transformer; the offline / cached path
falls back to a deterministic hash-based projection so that the benchmark and
unit tests never hit the network.
"""
from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np


class TextEncoder:
    def __init__(
        self,
        backbone: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_offline: bool = True,
        dim: int = 128,
    ) -> None:
        self.backbone = backbone
        self.dim = dim
        self._model = None
        self._offline = cache_offline
        if not cache_offline:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(backbone)
                self.dim = self._model.get_sentence_embedding_dimension()
            except Exception:
                self._offline = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._offline and self._model is not None:
            return np.asarray(self._model.encode(texts), dtype=np.float32)
        # Deterministic offline projection: hash each token, project into [-1,1].
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, tok in enumerate(t.split()[: self.dim]):
                h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16)
                out[i, j % self.dim] += ((h % 1000) / 500.0) - 1.0
            n = float(np.linalg.norm(out[i])) + 1e-9
            out[i] /= n
        return out

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
