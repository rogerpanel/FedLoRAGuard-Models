"""Weight-modality features for an adapter (Section 4.1).

Two paths are exposed:

* :func:`spectral_signature` -- exact SVD of the per-layer (B, A) update matrix
  truncated to the top-`k` singular values.  Used in the *Hidden mechanism*
  spectral analysis of Section 6.2.
* :func:`weight_features`    -- the production feature extractor: top-k
  singular values + per-layer Frobenius norms + a low-dimensional weight-
  difference fingerprint relative to a reference adapter.  Computed via one
  step of power iteration with a running average so that no full SVD is
  required per scan (this is the path tagged ``use_power_iteration: true`` in
  the YAML config).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _power_iteration_top_singular(
    M: np.ndarray, num_iter: int = 1, eps: float = 1e-9
) -> Tuple[float, np.ndarray, np.ndarray]:
    """One-step power iteration; returns (sigma_1, u, v)."""
    if M.size == 0:
        return 0.0, np.zeros(0), np.zeros(0)
    rng = np.random.default_rng(0)
    v = rng.normal(size=M.shape[1])
    v /= np.linalg.norm(v) + eps
    for _ in range(max(1, num_iter)):
        u = M @ v
        u_norm = np.linalg.norm(u) + eps
        u = u / u_norm
        v = M.T @ u
        v_norm = np.linalg.norm(v) + eps
        v = v / v_norm
        sigma = float(u @ M @ v)
    return sigma, u, v


def spectral_signature(
    BA: np.ndarray, topk: int = 8, use_power_iteration: bool = False, num_iter: int = 1
) -> np.ndarray:
    """Return the top-`topk` singular values of `BA = B @ A`."""
    if BA.size == 0:
        return np.zeros(topk, dtype=np.float32)
    if use_power_iteration:
        sigma, _, _ = _power_iteration_top_singular(BA, num_iter=num_iter)
        # tail values: estimate by repeatedly deflating the top component;
        # numerically inexact but cheap for the inline-scan path.
        sigmas = [sigma]
        residual = BA - sigma * np.outer(_power_iteration_top_singular(BA, num_iter)[1],
                                         _power_iteration_top_singular(BA, num_iter)[2])
        for _ in range(topk - 1):
            s, u, v = _power_iteration_top_singular(residual, num_iter=num_iter)
            sigmas.append(s)
            residual = residual - s * np.outer(u, v)
        out = np.array(sigmas[:topk], dtype=np.float32)
    else:
        # Exact SVD path -- numerically robust for benchmark generation.
        s = np.linalg.svd(BA, compute_uv=False)
        out = np.zeros(topk, dtype=np.float32)
        out[: min(topk, s.shape[0])] = s[: min(topk, s.shape[0])].astype(np.float32)
    return out


def weight_features(
    BA_per_layer: Dict[str, Tuple[np.ndarray, np.ndarray]],
    topk: int = 8,
    use_power_iteration: bool = True,
    reference_fp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Concatenate the spectral signature, per-layer Frobenius norms and a
    weight-difference fingerprint.  Output dim = ``topk + L + d_diff``.
    """
    sigs: List[np.ndarray] = []
    fros: List[float] = []
    for _, (B, A) in BA_per_layer.items():
        BA = B @ A
        sigs.append(spectral_signature(BA, topk=topk, use_power_iteration=use_power_iteration))
        fros.append(float(np.linalg.norm(BA)))
    if not sigs:
        return np.zeros(topk + 1, dtype=np.float32)
    sig_mean = np.mean(np.stack(sigs, axis=0), axis=0)
    fro = np.array(fros, dtype=np.float32)
    fro_summary = np.array([fro.mean(), fro.std(), fro.max(), fro.min()], dtype=np.float32)
    diff_fp = np.zeros(8, dtype=np.float32) if reference_fp is None else (sig_mean - reference_fp[:topk])[:8]
    return np.concatenate([sig_mean, fro_summary, diff_fp]).astype(np.float32)
