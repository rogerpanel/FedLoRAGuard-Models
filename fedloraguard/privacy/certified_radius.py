"""Theorem 2 of the paper -- Certified Poisoning Radius via DP composition.

Given the cumulative privacy budget epsilon_T and the top-two class
probabilities (p_hat_1, p_hat_2) emitted by the global verifier on a queried
adapter, we compute

    k* = floor( N * (Phi^{-1}(p_hat_1) - Phi^{-1}(p_hat_2)) / (2 e^{eps_T}) )

(Eq. (14)) -- the analytical lower bound on the number of colluding
malicious clients under which the global verdict is provably invariant.

The Cohen-style randomized-smoothing version ``cohen_certified_radius`` is
provided for the IDS-pipeline integration that needs an L2 certificate
expressed in feature space rather than client space.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from scipy.stats import norm  # type: ignore


@dataclass
class CertificateConfig:
    num_clients: int = 50
    failure_probability_budget: float = 1e-3
    smoothing_samples: int = 1000


def _phi_inv(p: float) -> float:
    p = float(min(max(p, 1e-9), 1.0 - 1e-9))
    return float(norm.ppf(p))


def certified_poisoning_radius(
    p_hat_1: float,
    p_hat_2: float,
    epsilon_T: float,
    num_clients: int,
) -> int:
    """Theorem 2 / Eq. (14)."""
    margin = _phi_inv(p_hat_1) - _phi_inv(p_hat_2)
    k_star = num_clients * margin / (2.0 * math.exp(epsilon_T))
    return max(0, int(math.floor(k_star)))


def cohen_certified_radius(p_hat_1: float, sigma: float) -> float:
    """L2 randomized-smoothing certificate (Cohen et al., 2019).

    Returned in the smoothing units used by the IDS pipeline (post-feature-space
    Gaussian); we expose it for the RobustIDPS.ai integration that uses it on
    the network IDS branch.
    """
    return sigma * _phi_inv(p_hat_1)


def confidence_lower_bound(
    successes: int, samples: int, alpha: float = 1e-3
) -> float:
    """Clopper-Pearson lower bound on the success probability."""
    from scipy.stats import beta  # type: ignore

    if successes == 0:
        return 0.0
    if successes == samples:
        return float(alpha ** (1.0 / samples))
    return float(beta.ppf(alpha, successes, samples - successes + 1))
